#!/usr/bin/env python3
"""Launch a native inference suite and its optional streaming metrics."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tailsteer.inference.experiment import prepare_inference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--suite", default="demos")
    p.add_argument("--output-dir")
    p.add_argument(
        "--policy", action="append", choices=("base", "tailsteer", "transition")
    )
    p.add_argument("--feedback", choices=("image", "latent"))
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    overrides = {"feedback": {"mode": a.feedback}} if a.feedback else {}
    plan = prepare_inference(
        a.run_dir,
        step=a.step,
        suite=a.suite,
        policies=a.policy,
        inference_overrides=overrides,
        output_dir=a.output_dir,
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={a.gpus}",
        str(ROOT / "scripts/inference_worker.py"),
        str(plan.config_path),
    ] + (["--resume"] if a.resume else [])
    print(json.dumps({**plan.summary(), "command": command}, indent=2), flush=True)
    if a.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    if "add_preserve" in plan.config["evaluation"]:
        from tailsteer.evaluation.add_preserve import run_with_metrics

        return run_with_metrics(
            command,
            cwd=ROOT,
            environment=env,
            config_path=plan.config_path,
            config=plan.config,
        )
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
