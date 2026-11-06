"""QIB prompt-only preparation and resumable local Q-Judger scoring.

Use the pinned official prompts, parser and aggregation with judge thinking enabled.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


from .common import PROJECT, digest, file_digest, read_json, write_json


def settings():
    from tailsteer.utils.config import load_config

    config = load_config(PROJECT / "configs/evaluation/qwenimagebench.yaml")
    config["python"] = os.environ.get("TAILSTEER_QIB_PYTHON", sys.executable)
    config["model"] = os.environ.get("TAILSTEER_QIB_MODEL", config["model"])
    return config


def runtime_environment(config):
    # Scope the judge's newer C++ runtime to its subprocesses only.
    lib = str(Path(config["python"]).parent.parent / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH")
    return {
        "LD_LIBRARY_PATH": lib + (":" + existing if existing else ""),
        "PYTHONPATH": str(PROJECT / "src"),
        "PYTHONNOUSERSITE": "1",
    }


def prepare(root: Path, source: Path | None = None):
    from .sources import prepare_code

    code = prepare_code(root, "qwenimagebench")
    raw = root / "datasets/qwenimagebench"
    raw.mkdir(parents=True, exist_ok=True)
    rows = {}
    if source:
        for lang in ("en", "zh"):
            values = [
                json.loads(line)
                for line in (source / "prompts" / f"qwenimagebench_{lang}.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            rows[lang] = {int(row["ID"]): row for row in values}
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "Qwen/Qwen-Image-Bench",
            "qwen_image_bench_hf_v0518.jsonl",
            repo_type="dataset",
            local_dir=raw,
        )
        values = [
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        for lang in ("en", "zh"):
            rows[lang] = {
                int(row["ID"]): {
                    "prompt": row["prompt_cn" if lang == "zh" else "prompt_en"]
                }
                for row in values
            }
    for lang in ("en", "zh"):
        if set(rows[lang]) != set(range(1, 1001)):
            raise ValueError(f"Qwen-Image-Bench {lang} requires IDs 1..1000")
    metadata = {
        int(r["ID"]): r for r in read_json(code / "metadata/bench_metadata.json")
    }
    if set(metadata) != set(range(1, 1001)):
        raise ValueError("QIB metadata IDs are incomplete")
    records = [
        {
            "id": f"qib-{lang}-{i:04d}",
            "official_id": [lang, i],
            "prompts": [rows[lang][i]["prompt"]],
            "metadata": {**metadata[i], "language": lang},
            "width": 1024,
            "height": 1024,
        }
        for i in range(1, 1001)
        for lang in ("en", "zh")
    ]
    manifest = {
        "schema_version": 1,
        "benchmark": "qwenimagebench",
        "task_type": "t2i",
        "records": records,
        "fingerprint": digest(records),
        "sessions": 2000,
        "turns": 2000,
        "raw_source": str(raw),
        "official_code": str(code),
    }
    path = root / "prepared/qwenimagebench/manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError(f"Prepared QIB identity changed: {path}")
    write_json(path, manifest)
    return path


def suite(tasks: Path, *, model_name=None):
    config = {
        "schema_version": 1,
        "name": "eval-qwenimagebench",
        "task_type": "t2i",
        "num_steps": 50,
        "tasks": str(tasks),
        "seed": 42,
        "backends": {
            "sensenova": {
                "schedule": {"shift_policy": "native"},
                "inference": {
                    "cfg_scale": 4.0,
                    "cfg_norm": "global",
                    "timestep_shift": 3.0,
                },
            },
            "flux2": {"schedule": {"shift_policy": "native"}},
        },
    }
    if model_name == "qwen_image_21":
        config["num_steps"] = 40
        config["backends"]["qwen21"] = {"schedule": {"shift_policy": "native"}}
    elif model_name == "flux1_kontext_dev":
        config["num_steps"] = 30
        config["backends"]["kontext"] = {
            "schedule": {"shift_policy": "native"},
            "inference": {"conditioning": "text_only_no_reference_v1"},
        }
    return config


def _upstream(code):
    sys.path.insert(0, str(code))
    return importlib.import_module("judge")


def validate_dimension(value, dim):
    from score_utils import CHECKLIST_L3_TO_L2

    if (
        not isinstance(value, dict)
        or not value
        or any(not isinstance(v, dict) for v in value.values())
    ):
        raise ValueError(f"Empty/malformed QIB dimension: {dim}")
    flattened = {k: v for children in value.values() for k, v in children.items()}
    expected = set(CHECKLIST_L3_TO_L2[dim])
    if set(flattened) != expected or len(flattened) != sum(map(len, value.values())):
        raise ValueError(
            f"Incomplete QIB checklist: {dim}; missing={sorted(expected - set(flattened))}"
        )
    for entry in flattened.values():
        score = entry.get("score") if isinstance(entry, dict) else entry
        if type(score) is int and score in (0, 1, 2):
            continue
        if isinstance(score, str) and score.upper() == "N/A":
            continue
        raise ValueError(f"Invalid QIB score: {score!r}")


def validate_scores(scores, dims, upstream):
    expected = upstream.parse_dims_by_level1(dims)
    if not isinstance(scores, dict) or set(scores) != set(expected):
        raise ValueError("Missing/extra QIB judge dimensions")
    for dim in expected:
        validate_dimension(scores[dim], dim)


def retry_invalid_dimensions(judge, row, record, upstream):
    """Retry only malformed dimensions, keeping valid scores and original replies.

    Greedy decoding repeats omitted N/A groups on a plain resume. Clarify only
    JSON completeness, with the same image, rubric and judge decoding settings.
    Never infer omitted scores from reasoning text or fill missing keys as N/A.
    """
    scores = record["scores"]
    if not isinstance(scores, dict):
        record["scores"] = scores = {}
    invalid = []
    for dim in upstream.parse_dims_by_level1(row["metadata"]["dims_en"]):
        try:
            validate_dimension(scores.get(dim), dim)
        except ValueError:
            invalid.append(dim)
    if not invalid:
        return
    from score_utils import CHECKLIST_L3_TO_L2

    with upstream.load_and_resize_image(row["outputs"][0]) as image:
        for dim in invalid:
            suffix = (
                "\n\n# JSON completeness requirement\n"
                "Return every criterion in the checklist, including every criterion "
                "whose score is N/A. Do not omit a criterion or an entire group, "
                "even if all its scores are N/A. Keep the scoring rules unchanged.\n"
                "Required criterion names: " + json.dumps(list(CHECKLIST_L3_TO_L2[dim]))
            )
            prompt = upstream.USER_PROMPT_TEMPLATE.format(
                prompt=row["prompts"][0],
                level1_dim=dim,
                format_checklist=upstream.DIM_TO_CHECKLIST[dim],
            )
            for attempt in range(1, 3):
                task = {
                    "system_prompt": upstream.SYSTEM_PROMPT,
                    "user_text": prompt + suffix,
                    "image": image,
                    "level1_dim": dim,
                    "row_id": row["id"],
                }
                (raw,) = judge.generate_batch([task])
                value = upstream._parse_output_to_scores(raw, dim)
                audit = {
                    "dimension": dim,
                    "attempt": attempt,
                    "prompt_suffix": suffix,
                    "raw": raw,
                    "scores": value,
                }
                record.setdefault("format_retries", []).append(audit)
                try:
                    validate_dimension(value, dim)
                except ValueError as error:
                    audit["error"] = str(error)
                    # A different explicit schema makes the bounded second retry
                    # useful even though the judge uses deterministic decoding.
                    schema = {}
                    for criterion, group in CHECKLIST_L3_TO_L2[dim].items():
                        schema.setdefault(group, {})[criterion] = {
                            "score": "0, 1, 2, or N/A"
                        }
                    suffix += (
                        "\nUse this complete key structure (replace every score placeholder):\n"
                        + json.dumps(schema)
                    )
                else:
                    scores[dim] = value
                    print(
                        f"QIB recovered {row['id']} / {dim}, format attempt {attempt}",
                        flush=True,
                    )
                    break


def _cached(row, work, upstream):
    path = work / "items" / f"{row['id']}.json"
    if not path.exists():
        return None
    data = read_json(path)
    if data["input_digest"] != digest(row):
        raise ValueError(f"QIB cached input changed: {path}")
    validate_scores(data["scores"], row["metadata"]["dims_en"], upstream)
    return data


def score(code, bundle, work, workers):
    config = settings()
    upstream = _upstream(code)
    identity = {
        "settings": config,
        "weights_index": file_digest(
            Path(config["model"]) / "model.safetensors.index.json"
        ),
        "upstream": read_json(code / "source.json"),
    }
    receipt = work / "qjudger.json"
    if receipt.exists() and read_json(receipt) != identity:
        raise ValueError("Q-Judger identity changed; use a separate result directory")
    write_json(receipt, identity)
    pending = [r for r in bundle["records"] if _cached(r, work, upstream) is None]
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    count = min(workers, len(devices), len(pending))
    children = []
    try:
        for rank in range(count):
            shard = work / f"shard-{rank}.json"
            write_json(
                shard,
                {
                    "records": pending[rank::count],
                    "config": config,
                    "code": str(code),
                    "work": str(work),
                },
            )
            log = (work / f"judge-rank-{rank:02d}.log").open("a")
            env = {
                **os.environ,
                **runtime_environment(config),
                "CUDA_VISIBLE_DEVICES": devices[rank],
            }
            process = subprocess.Popen(
                [config["python"], "-u", "-m", __name__, str(shard)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append((process, log))
        failed = [p.pid for p, _ in children if p.wait() != 0]
        if failed:
            raise RuntimeError(
                f"Q-Judger shards failed: {failed}; inspect {work}/judge-rank-*.log"
            )
    finally:
        for process, log in children:
            if process.poll() is None:
                process.terminate()
            log.close()
    results = {lang: [] for lang in ("en", "zh")}
    recovered = 0
    for row in bundle["records"]:
        item = _cached(row, work, upstream)
        if item is None:
            raise ValueError(f"Missing QIB score: {row['id']}")
        results[row["metadata"]["language"]].append(item["scores"])
        recovered += bool(item.get("format_retries"))
    return {
        "judge": "Q-Judger",
        "count": len(bundle["records"]),
        "per_language": {
            lang: upstream.compute_bench_scores(items)
            for lang, items in results.items()
            if items
        },
        "format_recovered_items": recovered,
        "note": "Official QIB aggregation; EN and ZH reported separately. Malformed dimensions receive an audited JSON-completeness retry; no missing scores are imputed.",
    }


def create_judge(config):
    from backends.ms_swift_backend import MsSwiftJudge
    from swift import InferRequest

    class ThinkingJudge(MsSwiftJudge):
        """Keep the official decoding settings and explicitly enable thinking."""

        def generate_batch(self, items):
            requests = [
                InferRequest(
                    messages=[
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["user_text"]},
                    ],
                    images=[item["image"]],
                    chat_template_kwargs={"enable_thinking": True},
                )
                for item in items
            ]
            responses = self.engine.infer(requests, self.request_config)
            return [r.choices[0].message.content for r in responses]

    return ThinkingJudge(
        config["model"], max_batch_size=config["batch_size"], max_new_tokens=4096
    )


def worker(path):
    import pandas as pd

    # Q-Judger uses ordinary BF16 layers; an optional FP8 engine is unnecessary.
    sys.modules["transformer_engine"] = None
    spec = read_json(path)
    config, work = spec["config"], Path(spec["work"])
    upstream = _upstream(Path(spec["code"]))
    judge = create_judge(config)
    args = SimpleNamespace(
        batch_size=config["batch_size"],
    )
    records = spec["records"]
    failures = []

    def save_result(row, record):
        retry_invalid_dimensions(judge, row, record, upstream)
        failed_path = work / "failed" / f"{row['id']}.json"
        try:
            validate_scores(record["scores"], row["metadata"]["dims_en"], upstream)
        except ValueError as error:
            failures.append(row["id"])
            record["error"] = str(error)
            write_json(failed_path, record)
        else:
            record.pop("error", None)
            write_json(work / "items" / f"{row['id']}.json", record)
            failed_path.unlink(missing_ok=True)

    fresh = []
    for row in records:
        failed_path = work / "failed" / f"{row['id']}.json"
        if failed_path.exists():
            record = read_json(failed_path)
            if record["input_digest"] != digest(row):
                raise ValueError(f"QIB failed input changed: {failed_path}")
            save_result(row, record)
        else:
            fresh.append(row)
    for offset in range(0, len(fresh), config["batch_size"]):
        chunk = fresh[offset : offset + config["batch_size"]]
        inputs = pd.DataFrame(
            [
                {"ID": i, "prompt": r["prompts"][0], "image_path": r["outputs"][0]}
                for i, r in enumerate(chunk)
            ]
        )
        meta = pd.DataFrame(
            [
                {"ID": i, "dims_en": r["metadata"]["dims_en"]}
                for i, r in enumerate(chunk)
            ]
        )
        outputs, _, scores, _ = upstream._run_batch_inference(
            judge, args, inputs, meta, "QIB"
        )
        for row, raw, value in zip(chunk, outputs, scores, strict=True):
            record = {"input_digest": digest(row), "scores": value, "raw": raw}
            save_result(row, record)
        print(
            f"QIB processed {len(records) - len(fresh) + offset + len(chunk)}/{len(records)}; failed={len(failures)}",
            flush=True,
        )
    if failures:
        raise RuntimeError(f"Incomplete QIB judgments: {failures}; resume retries them")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    worker(parser.parse_args().shard)
