#!/usr/bin/env python3
"""Build a local trajectory cache for one official image-edit model."""

from __future__ import annotations

import argparse
import json
import os

from tailsteer.cache import build_trajectory_cache, validate_trajectory_config
from tailsteer.cache.extend import extend_trajectory_points, inspect_point_extension
from tailsteer.utils import (
    cache_root,
    data_root,
    load_config,
    model_root,
    run_root,
    workspace_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--target-accepted", type=int)
    parser.add_argument("--candidate-end", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--task-shard-id", type=int)
    parser.add_argument("--num-task-shards", type=int)
    parser.add_argument("--disable-filter", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--extend-points",
        action="store_true",
        help="Add states to an existing grouped cache, retaining old tensors",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.target_accepted is not None:
        config["cache"]["target_accepted"] = args.target_accepted
    if args.candidate_end is not None:
        config["cache"]["candidate_end"] = args.candidate_end
    if args.output_dir is not None:
        config["cache"]["output_dir"] = args.output_dir
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    if args.task_shard_id is not None:
        config["cache"]["task_shard_id"] = args.task_shard_id
    if args.num_task_shards is not None:
        config["cache"]["num_task_shards"] = args.num_task_shards
    if args.disable_filter:
        config["cache"].setdefault("filter", {})["enabled"] = False
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"Consistent Editing workspace: {workspace_root()}")
        print(f"TailSteer data root: {data_root()}")
        print(f"TailSteer cache root: {cache_root()}")
        print(f"TailSteer model root: {model_root()}")
        print(f"TailSteer run root: {run_root()}")
    if args.validate_only:
        validate_trajectory_config(config)
        if args.extend_points:
            inspect_point_extension(config)
        if rank == 0:
            print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    result = (
        extend_trajectory_points if args.extend_points else build_trajectory_cache
    )(config)
    if rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
