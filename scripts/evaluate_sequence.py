#!/usr/bin/env python3
"""Run evaluation suites sequentially, optionally isolating benchmark failures."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tailsteer.evaluation.common import EDITING_BENCHMARKS, write_json
from evaluate_experiment import parse_args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--suite", action="append", type=Path)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--status-file", type=Path)
    args, common = parser.parse_known_args(argv)
    source_key = "run_dir"
    source_flag = "--" + source_key.replace("_", "-")
    runs = [p.resolve() for p in getattr(args, source_key)]
    if len(set(runs)) != len(runs):
        parser.error("Duplicate evaluation sources")
    for run in runs:
        config = run / "config.yaml"
        if not config.is_file():
            raise FileNotFoundError(config)
    results = []

    def save(status):
        if args.status_file:
            write_json(args.status_file, {"status": status, "results": results})

    for run in runs:
        for suite in args.suite or [None]:
            options = [source_flag, str(run)]
            if suite is not None:
                options += ["--suite", str(suite)]
            options += common
            selected = parse_args(options)
            benches = (
                (selected.bench or EDITING_BENCHMARKS)
                if args.continue_on_error
                else [None]
            )
            for bench in benches:
                # One subprocess per benchmark isolates generation, scoring and
                # plan failures, and releases its GPU memory before the next.
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("evaluate_experiment.py")),
                    *options,
                ]
                if bench is not None:
                    # Replace any repeated --bench options, not append to them.
                    command = [
                        *command[:2],
                        *without_benches(options),
                        "--bench",
                        bench,
                    ]
                entry = {
                    source_key: str(run),
                    "suite": str(suite) if suite else None,
                    "benchmark": bench,
                    "status": "running",
                    "command": command,
                }
                results.append(entry)
                save("running")
                print(
                    f"[evaluation sequence] START {run.name} {bench or suite}",
                    flush=True,
                )
                code = subprocess.run(command, check=False).returncode
                entry.update(
                    status="complete" if code == 0 else "failed", returncode=code
                )
                save("running")
                print(
                    f"[evaluation sequence] {entry['status'].upper()} {run.name} {bench or suite} exit={code}",
                    flush=True,
                )
                if code and not args.continue_on_error:
                    save("failed")
                    raise subprocess.CalledProcessError(code, command)
    failed = [entry for entry in results if entry["returncode"]]
    save("incomplete" if failed else "complete")
    if failed:
        print(
            f"{len(failed)}/{len(results)} evaluations failed; successful results retained. Use --resume to retry.",
            flush=True,
        )
        return 1
    return 0


def without_benches(options):
    result, skip = [], False
    for value in options:
        if skip:
            skip = False
        elif value == "--bench":
            skip = True
        elif not value.startswith("--bench="):
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
