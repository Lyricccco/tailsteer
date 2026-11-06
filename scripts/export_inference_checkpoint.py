#!/usr/bin/env python3
"""Create a portable inference export from an existing distributed checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from tailsteer.models import create_model
from tailsteer.training.checkpoint import (
    export_inference_model,
    load_model_checkpoint,
)
from tailsteer.training.distributed import initialize_distributed, prepare_model
from tailsteer.utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument(
        "--dtype",
        choices=("none", "float32", "float16", "bfloat16"),
        default="none",
        help=(
            "dtype for the portable weights; 'none' preserves the checkpoint "
            "dtype (use bfloat16 for SenseNova full-model exports)"
        ),
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    distributed = config.get("distributed", {"mode": "single"})
    context = initialize_distributed(
        str(distributed.get("mode", "single")),
        timeout_seconds=distributed.get("timeout_seconds"),
    )
    try:
        model = create_model(
            config["model"],
            device=context.device,
            include_vae=False,
            independent_actors=False,
            gradient_checkpointing=False,
        )
        model.configure_trainable(config["training"]["trainable"])
        root = prepare_model(model, context, distributed.get("fsdp2"))
        checkpoint = run_dir / "checkpoints" / f"step-{int(args.step):08d}"
        if not (checkpoint / "COMPLETE").is_file():
            raise FileNotFoundError(checkpoint)
        load_model_checkpoint(
            checkpoint,
            root=root,
            context=context,
            checkpoint_io=config["training"].get("checkpoint_io"),
        )
        tensor_dtype = None if args.dtype == "none" else getattr(torch, args.dtype)
        destination = export_inference_model(
            run_dir,
            name=f"step-{int(args.step):08d}",
            model=model,
            root=root,
            step=int(args.step),
            context=context,
            tensor_dtype=tensor_dtype,
        )
        if context.is_main:
            manifest = yaml.safe_load(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            print(
                {
                    "export": str(destination),
                    "model": manifest["model"],
                    "step": manifest["step"],
                    "weight_dtype": manifest.get("weight_dtype"),
                }
            )
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
