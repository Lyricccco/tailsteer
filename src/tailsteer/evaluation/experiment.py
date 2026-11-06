"""Evaluation plans and result I/O; forward passes remain entirely in inference."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import yaml

from .common import (
    PROJECT,
    atomic_text,
    contained,
    digest,
    file_digest,
    read_json,
    write_json,
)
from .preparation import verify_prepared


def benchmark_records(root: Path, benchmark: str, language="all", limit=None):
    """The same prepared selection for trained models and external baselines."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest_path = root / "prepared" / benchmark / "manifest.json"
    data = verify_prepared(manifest_path)
    records = data["records"]
    if language != "all":
        if benchmark != "gedit":
            raise ValueError("Language filtering is only defined for GEdit")
        records = [
            r for r in records if r["metadata"]["instruction_language"] == language
        ]
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("Empty evaluation selection")
    return manifest_path, data, records


def prepare_job(
    *,
    root: Path,
    benchmark: str,
    run: Path,
    step: int,
    policies=None,
    endpoint_modes=None,
    feedback="image",
    seed=0,
    limit=None,
    language="all",
    reuse=False,
    input_max_pixels=None,
) -> Path:
    if step <= 0 or (limit is not None and limit < 1):
        raise ValueError("step and limit must be positive")
    manifest_path, data, records = benchmark_records(root, benchmark, language, limit)
    run = run.resolve()
    config = yaml.safe_load((run / "config.yaml").read_text())
    from tailsteer.inference.registry import inference_backend_for_model

    backend = inference_backend_for_model(config["model"]["name"])
    num_steps = int(getattr(backend, "default_num_steps", 50))
    protocol = {
        "feedback": feedback,
        "seed": seed,
        "limit": limit,
        "language": language,
        "policies": policies,
        "endpoint_modes": endpoint_modes,
        "num_steps": num_steps,
    }
    t2i = data.get("task_type") == "t2i"
    if input_max_pixels is None and backend.key == "qwen21" and not t2i:
        input_max_pixels = 2048**2
    if backend.key == "kontext" and not t2i:
        from tailsteer.models.flux1_kontext.preprocessing import PREPROCESSING

        protocol["input_preprocessing"] = dict(PREPROCESSING)
        protocol["schedule"] = {
            "shift_policy": backend.default_shift_policy,
            "reference_height": 1024,
            "reference_width": 1024,
        }
    if input_max_pixels is not None:
        if t2i or backend.key != "qwen21" or int(input_max_pixels) < 1024:
            raise ValueError(
                "input_max_pixels currently requires Qwen editing and a budget >= 1024"
            )
        protocol["input_preprocessing"] = {
            "policy": "area_cap_lanczos_v1",
            "max_pixels": int(input_max_pixels),
        }
    if t2i:
        from .qwenimagebench import suite as qib_suite

        protocol["task_type"] = "t2i"
        protocol["generation_settings"] = qib_suite(
            Path("tasks.json"), model_name=config["model"]["name"]
        )
        if benchmark != "qwenimagebench":
            protocol["generation_settings"]["name"] = f"eval-{benchmark}"
        if benchmark == "bizgeneval":
            protocol["image_geometry"] = "native_aspect_long_side1024_align32"
    label = f"{feedback}-{digest(protocol)[:10]}"
    output = (
        root
        / "results"
        / config["model"]["name"]
        / run.name
        / f"step-{step:08d}"
        / benchmark
        / label
    )
    plan = {
        "schema_version": 1,
        "benchmark": benchmark,
        "run_dir": str(run),
        "step": step,
        "training_config_sha256": file_digest(run / "config.yaml"),
        "dataset_fingerprint": data["fingerprint"],
        "prepared": str(manifest_path.parent),
        "eval_root": str(root),
        "protocol": protocol,
        "records": records,
        "output": str(output),
        "generation": str(output / "generation"),
        "official_code": data["official_code"],
        "subset": len(records) != len(data["records"]),
    }
    job = output / "job.json"
    if reuse and policies and not job.exists():
        existing = _reusable_generation_job(plan)
        if existing is not None:
            return existing
    if job.exists() and read_json(job) != plan:
        raise ValueError(
            f"Evaluation identity changed; choose another result root: {job}"
        )
    tasks = []
    for row in records:
        if t2i:
            tasks.append(
                {
                    "id": row["id"],
                    "prompt": row["prompts"][0],
                    "width": row["width"],
                    "height": row["height"],
                    "seed": seed + int(row["metadata"]["ID"]) - 1,
                }
            )
            continue
        tasks.append(
            {
                "id": row["id"],
                "image": row["image"],
                "chain_rounds": True,
                "rounds": [
                    {
                        "prompt": p,
                        "steps": num_steps,
                        "seed": (seed + int(digest(row["official_id"])[:8], 16) + i)
                        % (2**31),
                    }
                    for i, p in enumerate(row["prompts"])
                ],
            }
        )
    write_json(output / "tasks.json", tasks if t2i else {"tasks": tasks})
    suite = {
        "schema_version": 1,
        "name": f"eval-{benchmark}",
        "num_steps": num_steps,
        "seed": seed,
        "tasks": str(output / "tasks.json"),
        "image_root": str(manifest_path.parent),
        "backends": {"flux2": {"inference": {"decode_after_chain": "all_rounds"}}},
    }
    if t2i:
        suite = qib_suite(output / "tasks.json", model_name=config["model"]["name"])
        suite["name"] = f"eval-{benchmark}"
        suite["seed"] = seed
    if input_max_pixels is not None:
        suite["backends"][backend.key] = {
            "inference": {"input_max_pixels": int(input_max_pixels)}
        }
    if backend.key == "kontext" and not t2i:
        suite["backends"][backend.key] = {"schedule": protocol["schedule"]}
    atomic_text(output / "suite.yaml", yaml.safe_dump(suite, sort_keys=False))
    write_json(job, plan)
    return job


def _reusable_generation_job(plan: dict) -> Path | None:
    """Find an identical completed run with a superset of the requested policies."""
    selected = set(plan["protocol"]["policies"])
    for path in sorted(Path(plan["output"]).parent.glob("*/job.json")):
        candidate = read_json(path)
        keys = (
            "run_dir",
            "step",
            "training_config_sha256",
            "dataset_fingerprint",
            "records",
            "official_code",
            "prepared",
            "subset",
        )
        if any(candidate.get(k) != plan.get(k) for k in keys):
            continue
        protocol = candidate["protocol"]
        if not selected < set(protocol.get("policies") or []):
            continue
        if any(
            protocol.get(k) != v for k, v in plan["protocol"].items() if k != "policies"
        ):
            continue
        if generation_complete(candidate):
            return path
    return None


def prepare_policy_selection(job_path: Path, policies: list[str] | None) -> Path:
    """A report/scoring view; preserve original generation and scorer identities."""
    job = read_json(job_path)
    if policies is None or policies == job["protocol"].get("policies"):
        return job_path
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("Selected policies must be nonempty and unique")
    if not set(policies) <= set(collect_outputs(job)):
        raise ValueError("Selected policies are missing from existing generation")
    output = Path(job["output"]) / "policy-subsets" / digest(policies)[:10]
    selected = {
        **job,
        "output": str(output),
        "selected_policies": list(policies),
        "score_source": str(job_path.resolve()),
    }
    path = output / "job.json"
    if path.exists() and read_json(path) != selected:
        raise ValueError(f"Policy selection identity changed: {path}")
    write_json(path, selected)
    return path


def score_directory(job: dict, policy: str) -> Path:
    source = read_json(Path(job["score_source"])) if job.get("score_source") else job
    return Path(source["output"]) / "scores" / policy


def generation_command(
    job: dict, *, python: str, gpus: int, resume: bool, dry_run=False
) -> list[str]:
    command = [
        python,
        str(PROJECT / "scripts/inference_experiment.py"),
        "--run-dir",
        job["run_dir"],
        "--step",
        str(job["step"]),
        "--suite",
        str(Path(job["output"]) / "suite.yaml"),
        "--output-dir",
        job["generation"],
        "--gpus",
        str(gpus),
        "--feedback",
        job["protocol"]["feedback"],
    ]
    if job["protocol"].get("task_type") == "t2i":
        command = command[:-2]  # No source-image feedback in T2I.
    for policy in job["protocol"]["policies"] or []:
        command += ["--policy", policy]
    for mode in job["protocol"]["endpoint_modes"] or []:
        command += ["--endpoint-mode", mode]
    if resume:
        command.append("--resume")
    if dry_run:
        command.append("--dry-run")
    return command


def copy_asset(source: Path, destination: Path) -> None:
    """Byte-identical copies; scorers never mutate generation files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_digest(source) != file_digest(destination):
            raise ValueError(f"Stale evaluation image: {destination}")
        return
    shutil.copyfile(source, destination)


def prepare_score_subset(job_path: Path, limit: int) -> Path:
    """Reuse completed sample chains, leaving the parent generation plan untouched."""
    if limit < 1:
        raise ValueError("Score limit must be positive")
    job = read_json(job_path)
    if job.get("score_subset_of"):
        raise ValueError("Select a scoring subset from its original generation job")
    output = Path(job["output"]) / "score-subsets" / f"first-{limit}"
    selected = {
        **job,
        "records": job["records"][:limit],
        "output": str(output),
        "subset": True,
        "score_subset_of": str(job_path.resolve()),
        "protocol": {**job["protocol"], "score_limit": limit},
    }
    collect_outputs(selected)  # Every selected sample must have all rounds/policies.
    path = output / "job.json"
    if path.exists() and read_json(path) != selected:
        raise ValueError(f"Scoring subset identity changed: {path}")
    write_json(path, selected)
    return path


def prepare_judge_variant(job_path: Path, settings: dict) -> Path:
    """Share generated images but never mix native and alternative judge scores."""
    job = read_json(job_path)
    output = Path(job["output"]) / "judge-variants" / digest(settings)[:10]
    selected = {**job, "output": str(output), "judge": settings}
    if "parser" in settings:
        selected["response_source"] = str(job_path.resolve())
    path = output / "job.json"
    if path.exists() and read_json(path) != selected:
        raise ValueError(f"Judge variant identity changed: {path}")
    write_json(path, selected)
    return path


def collect_outputs(
    job: dict, *, require_complete=True
) -> dict[str, dict[str, list[Path]]]:
    generation = Path(job["generation"])
    if (
        require_complete
        and not (generation / "COMPLETE").exists()
        and not job.get("score_subset_of")
    ):
        raise ValueError(f"Generation has not completed: {generation}")
    preprocessing = job["protocol"].get("input_preprocessing", {})
    if preprocessing.get("policy", "").startswith("kontext_"):
        generated = yaml.safe_load((generation / "config.yaml").read_text())
        if (
            generated["inference"].get("preprocessing") != preprocessing
            or generated["schedule"] != job["protocol"]["schedule"]
        ):
            raise ValueError(
                "Kontext generation preprocessing/schedule differs from evaluation"
            )
    policies = {}
    selection = job.get("selected_policies")
    expected = None
    for row in job["records"]:
        manifest = read_json(generation / "tasks" / row["id"] / "task.json")
        t2i = job["protocol"].get("task_type") == "t2i"
        if t2i and manifest["input"]["record"]["prompt"] != row["prompts"][0]:
            raise ValueError(f"Generated T2I prompt differs: {row['id']}")
        per_policy = {}
        for item in manifest["results"]:
            if t2i:
                item = {**item, "round": 1, "prompt": row["prompts"][0]}
            if selection is not None and item["policy"] not in selection:
                continue
            key = (item["policy"], int(item["round"]))
            if key in per_policy:
                raise ValueError(f"Duplicate generated result: {row['id']}, {key}")
            if item.get("prompt") != row["prompts"][key[1] - 1]:
                raise ValueError(
                    f"Generated prompt differs from benchmark: {row['id']}"
                )
            image = contained(generation, item["image"])
            if not image.is_file():
                raise FileNotFoundError(image)
            per_policy[key] = image
        names = {key[0] for key in per_policy}
        if (selection or job["protocol"].get("policies")) and names != set(
            selection or job["protocol"]["policies"]
        ):
            raise ValueError(f"Selected sample has incomplete policies: {row['id']}")
        if expected is not None and names != expected:
            raise ValueError("Inconsistent policy coverage across benchmark samples")
        expected = names
        for name in names:
            paths = [per_policy[(name, i + 1)] for i in range(len(row["prompts"]))]
            policies.setdefault(name, {})[row["id"]] = paths
    if not policies:
        raise ValueError("Benchmark has no generated policies")
    return policies


def generation_complete(job: dict) -> bool:
    """A marker alone is insufficient: require all selected policy/round images."""
    try:
        collect_outputs(job)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False
    return True


def export_policy(
    job: dict, policy: str, outputs: dict[str, list[Path]], destination: Path
) -> dict:
    """Adapt naming only; never resize generated images before an official scorer."""
    benchmark = job["benchmark"]
    prepared = Path(job["prepared"])
    source_root = destination / "sources"
    image_root = destination / "images"
    rows, checksums = [], {}
    for row in job["records"]:
        meta = dict(row["metadata"])
        source = contained(prepared, row["image"]) if "image" in row else None
        if benchmark == "qwenimagebench":
            targets = [image_root / meta["language"] / f"{meta['ID']}.png"]
        elif benchmark == "bizgeneval":
            targets = [image_root / f"{meta['prompt_id']:04d}.png"]
        elif benchmark == "imgedit-basic":
            copy_asset(source, contained(source_root, meta["id"]))
            targets = [contained(image_root, f"{meta['key']}.png")]
        elif benchmark == "gedit":
            targets = [
                contained(
                    image_root,
                    f"{meta['task_type']}/{meta['instruction_language']}/{meta['key']}.png",
                )
            ]
        elif benchmark == "rise":
            # Full official DataFrame contains this optional column; a smoke
            # subset may otherwise omit it entirely and break upstream indexing.
            meta.setdefault("reasoning_img", None)
            copy_asset(source, contained(source_root, meta["image"]))
            if "_reference_sha256" in meta:
                copy_asset(
                    contained(prepared / "references", meta["reference_img"]),
                    contained(source_root, meta["reference_img"]),
                )
                del meta["_reference_sha256"]
            targets = [contained(image_root, f"{meta['category']}/{meta['index']}.png")]
        elif benchmark == "mse":
            targets = [
                image_root / f"{meta['index']}_turn{i}_rep0_seed{meta['index']}.png"
                for i in range(5)
            ]
        elif benchmark == "edival":
            copy_asset(
                source,
                destination / "multipass" / f"{meta['index']}_input_raw{source.suffix}",
            )
            targets = [
                destination
                / "multipass"
                / f"{meta['index']}_input_raw_turn_{i + 1}.png"
                for i in range(3)
            ]
        else:
            raise ValueError(benchmark)
        for src, dst in zip(outputs[row["id"]], targets, strict=True):
            copy_asset(src, dst)
            checksums[str(dst.relative_to(destination))] = file_digest(dst)
        rows.append(
            {
                **row,
                "metadata": meta,
                "source": str(source),
                "outputs": [str(p) for p in targets],
            }
        )
    if benchmark == "rise":
        write_json(destination / "data.json", [r["metadata"] for r in rows])
    elif benchmark == "edival":
        csv_rows = [r for row in rows for r in row["metadata"]["rows"]]
        with (destination / "instructions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    bundle = {
        "benchmark": benchmark,
        "policy": policy,
        "records": rows,
        "images_sha256": checksums,
        "job": str(Path(job["output"]) / "job.json"),
    }
    if "judge" in job:
        bundle["judge"] = job["judge"]
    write_json(destination / "bundle.json", bundle)
    return bundle
