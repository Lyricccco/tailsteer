#!/usr/bin/env python3
"""Internal torchrun worker for registered model inference backends."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from tailsteer.inference.registry import run_registered_inference  # noqa: E402
from tailsteer.utils import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_registered_inference(config, resume=args.resume)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "product": result["product"],
                    "schema_version": result["schema_version"],
                    "backend": config["evaluation"]["backend"],
                    "task_count": len(result["tasks"]),
                    "policies": result.get(
                        "policy_order", list(result.get("policies", {}))
                    ),
                    "output_dir": config["run"]["output_dir"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
