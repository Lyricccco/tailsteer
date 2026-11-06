"""Isolated official scorer entry point (never imported by training)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .api import add_api_arguments, configure_api, redact_secret
from .api_retry import score_with_recovery
from .common import cache_environment, digest, load_sources, read_json, write_json
from .experiment import collect_outputs, export_policy, score_directory
from .scorers import JUDGES, SCORERS
from .sources import verify_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_api_arguments(parser)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--policy",
        help="Internal single-policy worker; defaults to all generated policies.",
    )
    args = parser.parse_args()
    job = read_json(args.job)
    root, benchmark = Path(job["eval_root"]), job["benchmark"]
    if benchmark == "edival":
        # vLLM 0.8's custom all-reduce exports CUDA IPC handles. Expandable
        # segments inherited from image generation cannot be exported this way.
        # Set this before importing/loading any CUDA judge, including workers.
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    os.environ.update(cache_environment(root))
    os.environ.pop("TRANSFORMERS_CACHE", None)
    code = verify_code(root, benchmark)
    if benchmark == "bizgeneval":
        for suffix in ("API_KEY", "BASE_URL"):
            if os.environ.get(f"BIZGEN_{suffix}"):
                os.environ[f"OPENAI_{suffix}"] = os.environ[f"BIZGEN_{suffix}"]
    if benchmark not in {"edival", "qwenimagebench"}:
        configure_api(args.api_key_file, args.api_base_url)
    if benchmark == "edival":
        from .judge_assets import verify_edival_judges

        verify_edival_judges(root)
    outputs = collect_outputs(job)
    if args.policy is None:
        # Release GPU judge engines and upstream module globals between policies.
        failures = []
        for policy in outputs:
            command = [
                sys.executable,
                "-m",
                "tailsteer.evaluation.scoring",
                "--job",
                str(args.job.resolve()),
                "--workers",
                str(args.workers),
                "--policy",
                policy,
            ]
            if args.resume:
                command.append("--resume")
            result = subprocess.run(command, check=False)
            if result.returncode:
                failures.append(policy)
                print(
                    f"Scoring incomplete for {policy}; continuing other policies.",
                    flush=True,
                )
        if failures:
            raise RuntimeError(
                f"Scoring incomplete for policies: {', '.join(failures)}; use --resume"
            )
        return
    if args.policy not in outputs:
        raise ValueError(f"Policy has no generated results: {args.policy}")
    outputs = {args.policy: outputs[args.policy]}
    original_cwd = Path.cwd()
    for policy, images in outputs.items():
        work = score_directory(job, policy)
        bundle = export_policy(job, policy, images, work)
        identity = {
            "dataset": job["dataset_fingerprint"],
            "protocol": job["protocol"],
            "images": bundle["images_sha256"],
            "judge": job["judge"]["model"] if "judge" in job else JUDGES[benchmark],
            "api_base": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if benchmark not in {"edival", "qwenimagebench"}
            else None,
            "official_revision": load_sources()[benchmark]["code_revision"],
        }
        if "judge" in job:
            identity["judge_api"] = job["judge"]["api"]
            if "parser" in job["judge"]:
                identity["parser"] = job["judge"]["parser"]
        if benchmark == "qwenimagebench":
            from .qwenimagebench import settings
            from .common import file_digest

            identity["qjudger"] = settings()
            identity["qjudger_index_sha256"] = file_digest(
                Path(identity["qjudger"]["model"]) / "model.safetensors.index.json"
            )
        versions = {}
        for package in (
            "Pillow",
            "numpy",
            "pandas",
            "openai",
            "transformers",
            "torch",
            "vllm",
        ):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                continue
        identity["runtime_versions"] = versions
        if benchmark == "edival":
            identity["judge_assets"] = read_json(root / "models/edival-models.json")
            identity["groundingdino_weights"] = read_json(
                root / "models/groundingdino/source.json"
            )
        receipt = work / "scorer.json"
        if receipt.exists():
            if read_json(receipt) != identity:
                raise ValueError(f"Scorer inputs/settings changed: {receipt}")
            if not args.resume:
                raise ValueError(f"Scoring already started: {work}; use --resume")
        write_json(receipt, identity)
        summary = work / "summary.json"
        if (
            args.resume
            and summary.exists()
            and read_json(summary).get("status") == "complete"
        ):
            continue
        os.chdir(work)
        try:
            if benchmark in {"edival", "qwenimagebench"}:
                metrics = SCORERS[benchmark](code, bundle, work, args.workers)
            else:
                metrics = score_with_recovery(
                    SCORERS[benchmark], code, bundle, work, args.workers
                )
            write_json(
                summary,
                {
                    "status": "complete",
                    "benchmark": benchmark,
                    "policy": policy,
                    "subset": job["subset"],
                    "sessions": len(job["records"]),
                    "identity": digest(identity),
                    "metrics": metrics,
                },
            )
        except Exception as error:
            write_json(
                summary,
                {
                    "status": "failed",
                    "benchmark": benchmark,
                    "policy": policy,
                    "error": redact_secret(str(error)),
                },
            )
            raise
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    main()
