"""Fixed-box preservation metrics, independent of model-specific generation code.

The optional CPU scorer watches round PNGs while the usual GPU runner generates.
All distances are to round zero. Pixels inside the allowed box are neutralized
before SSIM/LPIPS; spatial maps are then averaged over the exact outside mask.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import signal
import subprocess
import time
from pathlib import Path

from .common import atomic_text, digest, file_digest, read_json, write_json

PROTOCOL = {
    "version": 1,
    "reference": "white_box_input_round_zero",
    "mask": "exact_outside_human_box",
    "inside": "replace_prediction_with_reference",
    "psnr": "rgb_float01_masked_mse_cap120dB",
    "ssim": "gaussian11_sigma1.5_population_covariance_rgb_outside_map_mean",
    "lpips": "alex_v0.1_spatial_native_resolution_outside_map_mean",
    "aggregation": "equal_image_weight_population_std_ddof0",
}
METRICS = ("psnr", "ssim", "lpips")


def dataset_rows(config):
    settings = config["evaluation"]["add_preserve"]
    manifest_path = Path(settings["manifest"])
    manifest = read_json(manifest_path)
    tasks_path = Path(config["inference"]["tasks"])
    tasks = read_json(tasks_path)["tasks"]
    entries = {row["id"]: row for row in manifest["items"]}
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("add-preserve requires nonempty, unique task IDs")
    rows = []
    for task in tasks:
        row = dict(entries[task["id"]])
        expected = [{"prompt": r["prompt"], "seed": r["seed"]} for r in row["rounds"]]
        if (
            task["image"] != row["image"]
            or not task.get("chain_rounds")
            or not task["rounds"]
            or task["rounds"] != expected[: len(task["rounds"])]
        ):
            raise ValueError(
                f"Task differs from frozen add-preserve dataset: {task['id']}"
            )
        row["round_count"] = len(task["rounds"])
        rows.append(row)
    return manifest_path.parent, rows


def configure(config, settings):
    """Validate before generation, and include dataset identity in resume config."""
    inference = config["inference"]
    mode = inference.get("feedback", {}).get("mode")
    if mode == "latent" and config["evaluation"].get("backend") == "flux2":
        if (
            inference.get("decode_intermediate") is not False
            or inference.get("decode_after_chain") != "all_rounds"
        ):
            raise ValueError(
                "add-preserve latent evaluation requires all rounds decoded only after the chain"
            )
    elif mode != "image":
        raise ValueError(
            "add-preserve requires feedback=image or FLUX.2 feedback=latent"
        )
    if not config["inference"].get("policies"):
        raise ValueError("add-preserve requires ordinary inference policies")
    settings = dict(settings)
    settings.setdefault("python", sys.executable)
    settings.setdefault(
        "torch_home", os.environ.get("TORCH_HOME", str(Path.home() / ".cache/torch"))
    )
    config["evaluation"]["add_preserve"] = settings
    root, rows = dataset_rows(config)
    if Path(config["inference"]["image_root"]).resolve() != root.resolve():
        raise ValueError("add-preserve image_root must be the frozen dataset directory")
    for row in rows:
        for key in ("image", "keep_mask"):
            if file_digest(root / row[key]) != row[f"{key}_sha256"]:
                raise ValueError(f"Frozen asset changed: {row[key]}")
    settings.update(
        manifest_sha256=file_digest(Path(settings["manifest"])),
        tasks_sha256=file_digest(Path(config["inference"]["tasks"])),
        protocol=PROTOCOL,
    )
    if not Path(settings["python"]).is_file():
        raise FileNotFoundError(settings["python"])


class OutsideMetrics:
    def __init__(self, torch_home):
        import torch
        import lpips

        torch.set_num_threads(4)
        torch.hub.set_dir(str(Path(torch_home) / "hub"))
        self.network = lpips.LPIPS(
            net="alex", version="0.1", spatial=True, verbose=False
        )
        self.network.eval().requires_grad_(False)

    def __call__(self, reference, prediction, keep):
        import numpy as np
        import torch
        from skimage.metrics import structural_similarity

        if reference.shape != prediction.shape or reference.shape[:2] != keep.shape:
            raise ValueError(
                "Output dimensions changed; do not resize the evaluation mask"
            )
        if reference.ndim != 3 or reference.shape[2] != 3 or not keep.any():
            raise ValueError("Expected RGB arrays and a nonempty outside mask")
        reference = reference.astype(np.float32) / 255
        prediction = prediction.astype(np.float32) / 255
        mse = float(np.square(reference - prediction)[keep].mean(dtype=np.float64))
        psnr = -10 * np.log10(max(mse, 1e-12))
        prediction = np.where(keep[..., None], prediction, reference)
        _, ssim_map = structural_similarity(
            reference,
            prediction,
            data_range=1.0,
            channel_axis=2,
            full=True,
            gaussian_weights=True,
            sigma=1.5,
            win_size=11,
            use_sample_covariance=False,
        )

        def tensor(array):
            return (
                torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0) * 2 - 1
            )

        with torch.inference_mode():
            spatial = self.network(tensor(reference), tensor(prediction))[0, 0].numpy()
        values = {
            "psnr": float(psnr),
            "ssim": float(ssim_map[keep].mean(dtype=np.float64)),
            "lpips": float(spatial[keep].mean(dtype=np.float64)),
        }
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite preservation metrics: {values}")
        return values


def aggregate(records, rows, policies):
    import numpy as np

    result = []
    for policy in policies:
        for round_index in range(1, max(r["round_count"] for r in rows) + 1):
            expected = [r["id"] for r in rows if r["round_count"] >= round_index]
            values = [
                records[(task_id, policy, round_index)]["metrics"]
                for task_id in expected
                if (task_id, policy, round_index) in records
            ]
            result.append(
                {
                    "policy": policy,
                    "round": round_index,
                    "n": len(values),
                    "expected": len(expected),
                    **{
                        f"{key}_{stat}": (
                            float(getattr(np, stat)([v[key] for v in values]))
                            if values
                            else None
                        )
                        for key in METRICS
                        for stat in ("mean", "std")
                    },
                }
            )
    return result


def write_report(destination, summary):
    """Shared report format for native models and API image-feedback runs."""
    summaries = summary["rounds"]
    write_json(destination / "summary.json", summary)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
    atomic_text(destination / "rounds.csv", buffer.getvalue())
    lines = [
        "# Add & Preserve",
        "",
        f"Scored: {summary['scored']}/{summary['expected']}; complete: {summary['complete']}.",
        "",
        "Outside the fixed box, relative to round zero. Mean ± population std across images.",
        "",
    ]
    if summary.get("api_protocol"):
        protocol = summary["api_protocol"]
        lines += [
            f"API input: `{protocol['input_transform']}`; native image feedback; "
            "predictions resized to the initial canvas for scoring only (Lanczos).",
            "",
        ]
    lines += [
        "| Policy | Round | Images | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        cells = [
            f"{item[k + '_mean']:.5f} ± {item[k + '_std']:.5f}" if item["n"] else "—"
            for k in METRICS
        ]
        lines.append(
            f"| {item['policy']} | {item['round']} | {item['n']}/{item['expected']} | "
            + " | ".join(cells)
            + " |"
        )
    atomic_text(destination / "report.md", "\n".join(lines) + "\n")


def score(config, *, status_path=None, poll_seconds=5, preservation_only=False):
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    settings = config["evaluation"]["add_preserve"]
    if settings["protocol"] != PROTOCOL:
        raise ValueError("Scoring protocol changed; use a new output directory")
    root, rows = dataset_rows(config)
    if (
        file_digest(Path(settings["manifest"])) != settings["manifest_sha256"]
        or file_digest(Path(config["inference"]["tasks"])) != settings["tasks_sha256"]
    ):
        raise ValueError("Frozen manifest/tasks changed after preparing inference")
    output = Path(config["run"]["output_dir"])
    destination = output / "add-preserve"
    identity = digest({"settings": settings, "inference": config["inference"]})
    policies = list(config["inference"]["policies"])
    network = OutsideMetrics(settings["torch_home"])
    if status_path:
        status_path.with_suffix(".ready").touch()
        # The generation runner owns creation/validation of its result directory.
        while not (output / "config.yaml").is_file():
            if read_json(status_path)["state"] == "finished":
                return 1
            time.sleep(1)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "COMPLETE").unlink(missing_ok=True)
    records, seen = {}, {}
    last_count = -1
    while True:
        finished = status_path is None or read_json(status_path)["state"] == "finished"
        failures = {}
        changed = False
        for row in rows:
            reference = keep = None
            for policy in policies:
                for round_index in range(1, row["round_count"] + 1):
                    key = (row["id"], policy, round_index)
                    path = (
                        output
                        / "tasks"
                        / row["id"]
                        / policy
                        / f"round-{round_index:02d}.png"
                    )
                    receipt = (
                        destination
                        / "records"
                        / f"{row['id']}-{policy}-{round_index:02d}.json"
                    )
                    if not path.is_file():
                        changed |= records.pop(key, None) is not None
                        seen.pop(key, None)
                        continue
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        records.pop(key, None)
                        seen.pop(key, None)
                        continue  # The runner can replace an incomplete task on resume.
                    stamp = (stat.st_size, stat.st_mtime_ns)
                    if not finished and seen.get(key) == stamp:
                        continue
                    try:
                        image_hash = file_digest(path)
                        saved = read_json(receipt) if receipt.is_file() else None
                        if (
                            saved
                            and saved["identity"] == identity
                            and saved["image_sha256"] == image_hash
                        ):
                            records[key] = saved
                            seen[key] = stamp
                            continue
                        if reference is None:
                            for field in ("image", "keep_mask"):
                                if (
                                    file_digest(root / row[field])
                                    != row[f"{field}_sha256"]
                                ):
                                    raise ValueError("Frozen input or mask changed")
                            with Image.open(root / row["image"]) as im:
                                reference = np.array(im.convert("RGB"))
                            with Image.open(root / row["keep_mask"]) as im:
                                keep = np.array(im.convert("L")) > 0
                            with Image.open(
                                output / "tasks" / row["id"] / "source.png"
                            ) as im:
                                actual = np.array(im.convert("RGB"))
                            if not np.array_equal(actual, reference):
                                raise ValueError(
                                    "Model input differs from benchmark reference"
                                )
                        with Image.open(path) as im:
                            prediction = np.array(im.convert("RGB"))
                        values = network(reference, prediction, keep)
                        # A runner may be rewriting an incomplete chain during resume.
                        if file_digest(path) != image_hash:
                            continue
                        record = {
                            "id": row["id"],
                            "category": row["category"],
                            "policy": policy,
                            "round": round_index,
                            "identity": identity,
                            "image_sha256": image_hash,
                            "reference_sha256": row["image_sha256"],
                            "mask_sha256": row["keep_mask_sha256"],
                            "metrics": values,
                        }
                        write_json(receipt, record)
                        records[key] = record
                        seen[key] = stamp
                        changed = True
                    except (OSError, UnidentifiedImageError, ValueError) as error:
                        records.pop(key, None)
                        seen.pop(key, None)
                        failures["/".join(map(str, key))] = str(error)
            # Publish at least once per completed image, without waiting for all tasks.
            if changed or finished or len(records) != last_count:
                summaries = aggregate(records, rows, policies)
                expected = sum(r["round_count"] for r in rows) * len(policies)
                complete = finished and len(records) == expected and not failures
                summary = {
                    "protocol": PROTOCOL,
                    "dataset": str(root),
                    "identity": identity,
                    "complete": complete,
                    "scored": len(records),
                    "expected": expected,
                    "rounds": summaries,
                    "errors": failures,
                }
                write_report(destination, summary)
                last_count = len(records)
                changed = False
        print(
            f"add-preserve scored={len(records)}/{expected} complete={complete}",
            flush=True,
        )
        if finished:
            if complete:
                (destination / "COMPLETE").touch()
            else:
                (destination / "COMPLETE").unlink(missing_ok=True)
            # Separate receipts keep existing pixel metrics resumable and auditable.
            # The API scorer runs on CPU after generation; no inference code changes.
            object_code = 0
            if not preservation_only:
                from .add_preserve_objects import score as score_objects

                object_code = score_objects(config)
            return (0 if complete else 1) or object_code
        time.sleep(poll_seconds)


def run_with_metrics(command, *, cwd, environment, config_path, config):
    """Own both children, propagate failure, and keep existing runners untouched."""
    import yaml

    result = Path(config["run"]["output_dir"])
    if result.exists() and any(result.iterdir()) and "--resume" not in command:
        raise FileExistsError(f"inference output exists; pass --resume: {result}")
    saved_config = result / "config.yaml"
    if saved_config.is_file():
        saved = yaml.safe_load(saved_config.read_text())
        # Workers record load_config's source path; the launcher has not loaded
        # that file. This provenance key is not an inference setting.
        expected = dict(config)
        saved.pop("_config_path", None)
        expected.pop("_config_path", None)
        if saved != expected:
            raise ValueError("resume config differs from saved inference config")
    settings = config["evaluation"]["add_preserve"]
    output = config_path.parent / "add-preserve-runtime"
    output.mkdir(parents=True, exist_ok=True)
    status = output / f"generation-{os.getpid()}.json"
    ready = status.with_suffix(".ready")
    write_json(status, {"state": "running"})
    env = environment.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        TORCH_HOME=settings["torch_home"],
        PYTHONNOUSERSITE="1",
        PYTHONPATH=str(cwd / "src"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
    )
    processes = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / "scorer.log").open("a") as log:
            scorer = subprocess.Popen(
                [
                    settings["python"],
                    "-u",
                    str(cwd / "scripts/score_add_preserve.py"),
                    str(config_path),
                    "--generation-status",
                    str(status),
                ],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(scorer)
            print(f"Live preservation scoring log: {output / 'scorer.log'}", flush=True)
            deadline = time.monotonic() + 180
            while not ready.exists():
                if scorer.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Metric scorer failed to initialize; see {output / 'scorer.log'}"
                    )
                time.sleep(1)
            generation = subprocess.Popen(
                command, cwd=cwd, env=environment, start_new_session=True
            )
            processes.append(generation)
            while generation.poll() is None:
                if scorer.poll() is not None:
                    raise RuntimeError(
                        f"Metric scorer stopped; see {output / 'scorer.log'}"
                    )
                time.sleep(2)
            write_json(
                status, {"state": "finished", "exit_code": generation.returncode}
            )
            scoring_code = scorer.wait()
            return generation.returncode or scoring_code
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        for child in reversed(processes):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        ready.unlink(missing_ok=True)
