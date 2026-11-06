#!/usr/bin/env python3
"""Score existing add-preserve images, optionally following a live generation."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tailsteer.evaluation.add_preserve import score
from tailsteer.evaluation.common import write_json
from tailsteer.evaluation.api import redact_secret
from tailsteer.utils.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--generation-status", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--objects-only",
        action="store_true",
        help="CPU API judging of saved crops only",
    )
    mode.add_argument(
        "--preservation-only", action="store_true", help="Only PSNR/SSIM/LPIPS"
    )
    parser.add_argument(
        "--policy", action="append", help="Object-only policy selection; repeatable"
    )
    parser.add_argument(
        "--object-batch",
        type=Path,
        help="YAML entries: [{config: path, policies: [...]}]; continue after failures",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate object batch without API calls"
    )
    args = parser.parse_args()
    if args.object_batch:
        if (
            args.config
            or args.generation_status
            or args.preservation_only
            or args.policy
        ):
            parser.error("--object-batch is a standalone object scoring command")
        from tailsteer.evaluation.add_preserve_objects import score as score_objects
        from tailsteer.evaluation.add_preserve import dataset_rows

        batch = load_config(args.object_batch)
        results = []
        for entry in batch["entries"]:
            item = {"config": entry["config"], "policies": entry.get("policies")}
            try:
                config = load_config(entry["config"])
                _, rows = dataset_rows(config)
                policies = entry.get("policies") or list(
                    config["inference"]["policies"]
                )
                if not set(policies) <= set(config["inference"]["policies"]):
                    raise ValueError("Unknown policy")
                expected = sum(r["round_count"] for r in rows) * len(policies)
                missing = [
                    str(
                        Path(config["run"]["output_dir"])
                        / "tasks"
                        / r["id"]
                        / p
                        / f"round-{t:02d}.png"
                    )
                    for r in rows
                    for p in policies
                    for t in range(1, r["round_count"] + 1)
                    if not (
                        Path(config["run"]["output_dir"])
                        / "tasks"
                        / r["id"]
                        / p
                        / f"round-{t:02d}.png"
                    ).is_file()
                ]
                if args.dry_run and missing:
                    raise ValueError(
                        f"{len(missing)} images missing, e.g. {missing[0]}"
                    )
                print(
                    f"Object scoring: {entry['config']} policies={policies} expected={expected}",
                    flush=True,
                )
                item.update(
                    expected=expected,
                    exit_code=0
                    if args.dry_run
                    else score_objects(config, policies=policies),
                )
            except Exception as error:
                item.update(exit_code=1, error=redact_secret(str(error)))
                print(f"Object scoring failed; continuing: {item['error']}", flush=True)
            results.append(item)
            write_json(
                args.object_batch.with_suffix(
                    ".dry-run.json" if args.dry_run else ".status.json"
                ),
                {
                    "complete": len(results) == len(batch["entries"])
                    and all(r["exit_code"] == 0 for r in results),
                    "entries": results,
                },
            )
        raise SystemExit(int(any(r["exit_code"] for r in results)))
    if not args.config:
        parser.error("config is required")
    if args.dry_run or (args.policy and not args.objects_only):
        parser.error(
            "--dry-run requires --object-batch; --policy requires --objects-only"
        )
    config = load_config(args.config)
    if args.objects_only:
        if args.generation_status:
            parser.error("--objects-only scores saved images, not live generation")
        from tailsteer.evaluation.add_preserve_objects import score as score_objects

        raise SystemExit(score_objects(config, policies=args.policy))
    raise SystemExit(
        score(
            config,
            status_path=args.generation_status,
            preservation_only=args.preservation_only,
        )
    )
