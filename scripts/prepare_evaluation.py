#!/usr/bin/env python3
"""Prepare our own copy of official editing benchmarks and pinned scorers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tailsteer.evaluation.common import (
    BENCHMARKS,
    EDITING_BENCHMARKS,
    DEFAULT_ROOT,
    cache_environment,
)
from tailsteer.evaluation.preparation import prepare_benchmark, verify_prepared
from tailsteer.evaluation.sources import prepare_code, prepare_dataset, verify_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bench", action="append", choices=BENCHMARKS)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Import one existing benchmark into our own prepared directory.",
    )
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify prepared assets and pinned scorers without downloading.",
    )
    assets = parser.add_mutually_exclusive_group()
    assets.add_argument(
        "--edival-judge-models",
        action="store_true",
        help="Also download the large EdiVal judges; requires HF access to DINOv3.",
    )
    assets.add_argument(
        "--edival-local-models",
        type=Path,
        metavar="DIRECTORY",
        help="Register existing EdiVal model folders offline, without copying weights.",
    )
    args = parser.parse_args()
    benches = args.bench or list(EDITING_BENCHMARKS)
    if (
        args.edival_judge_models or args.edival_local_models
    ) and "edival" not in benches:
        parser.error("EdiVal judge options require --bench edival (or all benches)")
    if args.source_dir and len(benches) != 1:
        parser.error("--source-dir requires exactly one --bench")
    root = args.eval_root.expanduser().resolve()
    os.environ.update(cache_environment(root))
    os.environ.pop("TRANSFORMERS_CACHE", None)
    for bench in benches:
        if args.verify:
            verify_code(root, bench)
            result = verify_prepared(root / "prepared" / bench / "manifest.json")
            print(
                f"{bench}: verified {result['sessions']} sessions, "
                f"{result['turns']} turns",
                flush=True,
            )
            continue
        if bench == "qwenimagebench":
            from tailsteer.evaluation.qwenimagebench import prepare

            print(prepare(root, args.source_dir), flush=True)
            continue
        if bench == "bizgeneval":
            from tailsteer.evaluation.bizgeneval import prepare

            print(prepare(root, args.source_dir), flush=True)
            continue
        code = prepare_code(root, bench)
        print(f"{bench}: official source {code}", flush=True)
        if not args.code_only:
            prepared = root / "prepared" / bench / "manifest.json"
            if prepared.exists() and args.source_dir is None:
                verify_code(root, bench)
                result = verify_prepared(prepared)
                print(f"{bench}: reusing {result['sessions']} prepared sessions")
                continue
            raw = (
                args.source_dir.resolve()
                if args.source_dir
                else prepare_dataset(root, bench)
            )
            print(prepare_benchmark(root, bench, raw, code), flush=True)
    if args.edival_judge_models or args.edival_local_models:
        from tailsteer.evaluation.judge_assets import (
            prepare_edival_judges,
            register_local_edival_judges,
            verify_edival_judges,
        )

        if args.verify:
            verify_edival_judges(root, hashes=True)
            print("EdiVal judge assets verified", flush=True)
        elif args.edival_local_models:
            print(
                json.dumps(
                    register_local_edival_judges(root, args.edival_local_models),
                    indent=2,
                )
            )
        else:
            prepare_edival_judges(root)


if __name__ == "__main__":
    main()
