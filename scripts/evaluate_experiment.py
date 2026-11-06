#!/usr/bin/env python3
"""Generate and score editing benchmarks for a training run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tailsteer.evaluation.api import add_api_arguments, redact_secret
from tailsteer.evaluation.common import (
    BENCHMARKS,
    EDITING_BENCHMARKS,
    DEFAULT_ROOT,
    PROJECT,
    TRAINING_PYTHON,
    cache_environment,
    read_json,
    write_json,
)
from tailsteer.evaluation.experiment import (
    collect_outputs,
    generation_command,
    generation_complete,
    prepare_job,
    prepare_judge_variant,
    prepare_policy_selection,
    prepare_score_subset,
    score_directory,
)
from tailsteer.evaluation.judge_config import add_judge_argument, load_judge_config
from tailsteer.evaluation.suites import load_suite, suite_defaults


def parse_args(argv=None):
    suite_parser = argparse.ArgumentParser(add_help=False)
    suite_parser.add_argument("--suite", type=Path)
    selected, _ = suite_parser.parse_known_args(argv)
    defaults = suite_defaults(load_suite(selected.suite)) if selected.suite else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, help="Evaluation suite YAML; CLI overrides its defaults."
    )
    add_api_arguments(parser)
    add_judge_argument(parser)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--bench", action="append", choices=BENCHMARKS)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--stage", choices=("generate", "score", "report", "all"), default="generate"
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Inference policy: base, tailsteer, or transition.",
    )
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--input-max-pixels",
        type=int,
        help="Qwen editing only: cap source area, preserve aspect ratio.",
    )
    parser.add_argument(
        "--feedback",
        default="image",
        choices=("image", "latent", "continuous-image", "model-native"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke subset; always isolated from full benchmark output.",
    )
    parser.add_argument("--gedit-language", default="all", choices=("all", "en", "cn"))
    parser.add_argument("--training-python", default=TRAINING_PYTHON)
    parser.add_argument(
        "--scorer-python", help="Override the benchmark-specific isolated environment."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--score-limit",
        type=int,
        help="Score the first N existing samples separately, without regenerating.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # Explicit list options replace, rather than append to, the suite's lists.
    lists = {
        key: defaults.pop(key, None) for key in ("bench", "policy", "endpoint_mode")
    }
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.run_dir is not None and args.step is None:
        parser.error("--run-dir requires --step")
    if args.bench is None:
        args.bench = lists["bench"]
    if args.policy is None:
        args.policy = lists["policy"]
    if args.gpus < 1 or args.workers < 1:
        parser.error("--gpus and --workers must be positive")
    if args.score_limit is not None and (
        args.score_limit < 1 or args.stage not in {"score", "report"}
    ):
        parser.error(
            "--score-limit must be positive and used with --stage score or report"
        )
    return args


def _require_scores(job):
    for policy in collect_outputs(job):
        summary = score_directory(job, policy) / "summary.json"
        if not summary.is_file() or read_json(summary).get("status") != "complete":
            raise ValueError(f"Scoring incomplete: {summary}")


def _run_logged(command, output: Path, phase: str, *, env=None):
    """Keep worker tracebacks with the evaluation output."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{phase}.log").open("a", buffering=1) as log:
        log.write("\n$ " + shlex.join(command) + "\n")
        with subprocess.Popen(
            command,
            cwd=PROJECT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                    if phase == "score" and (
                        (
                            "Failed: Cuda error" in line
                            and "custom_all_reduce.cuh:" in line
                        )
                        or "EngineDeadError" in line
                        or "Engine core initialization failed" in line
                    ):
                        # A dead vLLM worker can leave its parent waiting forever.
                        # Fail this benchmark so --continue-on-error can advance.
                        raise RuntimeError(f"Scoring engine failed: {line.strip()}")
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    # Also terminate descendants that ignore SIGTERM after the
                    # immediate child has exited.
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise


def _run_job(args, generation_job, job_path):
    job = read_json(job_path)
    root = Path(job["eval_root"])
    phase = args.stage

    def status(value):
        if not args.dry_run:
            write_json(
                Path(job["output"]) / "pipeline_status.json",
                {
                    "status": value,
                    "phase": phase,
                    "requested_stage": args.stage,
                    "job": str(job_path),
                },
            )

    try:
        if args.stage in {"generate", "all"}:
            phase = "generate"
            status("running")
            if args.resume and generation_complete(generation_job):
                print(
                    "Generation complete; reusing images: "
                    + generation_job["generation"],
                    flush=True,
                )
            else:
                command = generation_command(
                    generation_job,
                    python=args.training_python,
                    gpus=args.gpus,
                    resume=args.resume,
                    dry_run=args.dry_run,
                )
                print(shlex.join(command), flush=True)
                if args.dry_run:
                    subprocess.run(command, check=True, cwd=PROJECT)
                else:
                    _run_logged(command, Path(generation_job["output"]), "generate")
            if not args.dry_run:
                collect_outputs(generation_job)
        if args.stage in {"score", "all"}:
            phase = "score"
            status("running")
            scorer = args.scorer_python or sys.executable
            if job["benchmark"] == "qwenimagebench" and not args.scorer_python:
                from tailsteer.evaluation.qwenimagebench import settings

                scorer = settings()["python"]
            if job["benchmark"] == "edival" and not args.scorer_python:
                scorer = os.environ.get("TAILSTEER_EDIVAL_PYTHON", scorer)
            command = [
                scorer,
                "-m",
                "tailsteer.evaluation.scoring",
                "--job",
                str(job_path),
                "--workers",
                str(args.workers),
            ]
            if job["benchmark"] == "qwenimagebench":
                command[-1] = str(args.gpus)
            if args.resume:
                command.append("--resume")
            if args.api_key_file:
                command += [
                    "--api-key-file",
                    str(args.api_key_file.expanduser().resolve()),
                ]
            if args.api_base_url:
                command += ["--api-base-url", args.api_base_url]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(PROJECT / "src")
                environment["PYTHONNOUSERSITE"] = "1"
                environment.update(cache_environment(root))
                environment.pop("TRANSFORMERS_CACHE", None)
                if job["benchmark"] == "qwenimagebench":
                    from tailsteer.evaluation.qwenimagebench import (
                        runtime_environment,
                        settings,
                    )

                    environment.update(runtime_environment(settings()))
                _run_logged(command, Path(job["output"]), "score", env=environment)
                _require_scores(job)
        if args.stage in {"report", "all"} and not args.dry_run:
            from tailsteer.evaluation.report import write_report

            phase = "report"
            status("running")
            print(write_report(Path(job["output"])), flush=True)
        status("complete")
    except Exception:
        status("failed")
        raise


def main(argv=None):
    args = parse_args(argv)
    root = args.eval_root.expanduser().resolve()
    judges = load_judge_config(args.judge_config)
    failures = []
    for bench in args.bench or EDITING_BENCHMARKS:
        selection = dict(
            root=root,
            benchmark=bench,
            policies=args.policy,
            feedback=args.feedback,
            seed=args.seed,
            limit=args.limit,
            language=args.gedit_language if bench == "gedit" else "all",
        )
        job_path = prepare_job(
            **selection,
            run=args.run_dir,
            step=args.step,
            reuse=args.resume,
            input_max_pixels=args.input_max_pixels,
        )
        job = read_json(job_path)
        if args.score_limit is not None:
            job_path = prepare_score_subset(job_path, args.score_limit)
            job = read_json(job_path)
        generation_job = job
        if args.stage != "generate" and bench in judges:
            job_path = prepare_judge_variant(job_path, judges[bench])
        job_path = prepare_policy_selection(job_path, args.policy)
        print(
            json.dumps(
                {"benchmark": bench, "job": str(job_path), "subset": job["subset"]}
            ),
            flush=True,
        )
        try:
            _run_job(args, generation_job, job_path)
        except Exception as error:
            status_path = Path(read_json(job_path)["output"]) / "pipeline_status.json"
            # Isolate scoring failures only. Generation/configuration failures
            # retain the existing fail-fast behavior.
            if (
                args.dry_run
                or not status_path.exists()
                or read_json(status_path)["phase"] != "score"
            ):
                raise
            failures.append(bench)
            print(
                f"{bench}: scoring incomplete; continuing remaining benchmarks. "
                f"{redact_secret(str(error))}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"Incomplete benchmark scoring: {', '.join(failures)}. "
            "Successful results are saved; use --resume to recover failed items."
        )


if __name__ == "__main__":
    main()
