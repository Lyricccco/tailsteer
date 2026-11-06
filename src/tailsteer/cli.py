"""Public preparation, cache, training, and inference commands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from .configuration import BACKBONES
from .utils.config import data_root, load_config, model_root

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "training-source": (
        "Lyricccco/TailSteer-Source",
        "1c6fc2c2f39e9b0facaadfbec70dc9d2763945f3",
        "training",
    ),
    "add-preserve": (
        "Lyricccco/Add-and-Preserve",
        "bf5afbbd6d1137b62d260122a5352ff127f81d69",
        "add-preserve",
    ),
}
SUITE_GROUPS = {
    "demos": [
        "cafe-20turn",
        "lace-dress-dance",
        "yard-object-composition",
        "people-composition",
    ],
    "t2i": ["qwenimagebench", "bizgeneval"],
    "single-turn": ["gedit", "rise", "imgedit-basic"],
    "multi-turn": ["mse", "edival"],
    "paper-benchmarks": [
        "qwenimagebench",
        "bizgeneval",
        "gedit",
        "rise",
        "imgedit-basic",
        "mse",
        "edival",
        "add-preserve",
    ],
}
BENCHMARKS = (
    "qwenimagebench",
    "bizgeneval",
    "gedit",
    "rise",
    "imgedit-basic",
    "mse",
    "edival",
)


def launch(script, arguments, *, gpus=None, dry_run=False):
    command = [sys.executable]
    if gpus is not None:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={gpus}",
        ]
    command += [str(ROOT / "scripts" / script), *map(str, arguments)]
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.call(command, cwd=ROOT, env=env)


def save_plan(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    config.pop("_config_path", None)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path.resolve()


def configuration(args, kind):
    path = args.config or ROOT / "configs" / kind / f"{args.backbone}.yaml"
    config = load_config(path)
    for key, value in (("path", args.model_path), ("code_path", args.model_code)):
        if value:
            config["model"][key] = str(Path(value).expanduser().resolve())
    return config


def prepare(args):
    """Download user-selected source data or a frozen benchmark snapshot."""
    if args.dataset == "model":
        from huggingface_hub import snapshot_download

        repo, folder = {
            "flux1-kontext-dev": (
                "black-forest-labs/FLUX.1-Kontext-dev",
                "FLUX.1-Kontext-dev",
            ),
            "flux2-klein-base-9b": (
                "black-forest-labs/FLUX.2-klein-base-9B",
                "FLUX.2-klein-base-9B",
            ),
            "qwen-image-2.1": ("Qwen/Qwen-Image-2.1", "Qwen-Image-2.1"),
            "sensenova-u1.5-8b-mot": (
                "sensenova/SenseNova-U1.5-8B-MoT",
                "SenseNova-U1.5-8B-MoT",
            ),
        }[args.backbone]
        snapshot_download(
            repo,
            revision=args.revision or "main",
            local_dir=args.output_dir or model_root() / folder,
        )
        return 0
    if args.dataset in DATASETS:
        from huggingface_hub import snapshot_download

        default_repo, revision, directory = DATASETS[args.dataset]
        repo = args.dataset_repo or default_repo
        output = Path(args.output_dir or data_root() / directory)
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=args.revision or (revision if repo == default_repo else "main"),
            local_dir=output,
            max_workers=32,
        )
        print(f"Downloaded {args.dataset} to {output.resolve()}")
        return 0
    return launch(
        "prepare_evaluation.py",
        ["--eval-root", args.eval_root, "--bench", args.dataset],
    )


def cache(args):
    config = configuration(args, "cache")
    output = Path(args.output_dir or config["cache"]["output_dir"]).resolve()
    config["cache"]["output_dir"] = str(output)
    if args.sources:
        config["data"]["manifest"] = str(Path(args.sources).resolve())
    if args.image_root:
        config["data"]["image_root"] = str(Path(args.image_root).resolve())
    # A source manifest contains one independent trajectory request per row.
    source = Path(config["data"]["manifest"])
    rows = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["source"] = row.pop("image", row.get("source"))
        if not row["source"]:
            raise ValueError("Every source row needs image, prompt, and a unique id")
        row["source_identity"] = row.pop(
            "source_id", row.get("source_identity", row["source"])
        )
        row["task"] = {"local": "local_edit", "identity": "noop"}.get(
            row.get("task"), row.get("task")
        )
        if row["task"] not in ("local_edit", "noop"):
            raise ValueError("Source task must be local or identity")
        rows.append(row)
    plan_dir = output.parent / ".plans" / output.name
    plan_dir.mkdir(parents=True, exist_ok=True)
    manifest = plan_dir / "sources.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    config["data"]["manifest"] = str(manifest)
    if args.limit:
        config["cache"]["candidate_end"] = args.limit
    if args.disable_filter:
        config["cache"]["filter"]["enabled"] = False
    plan = save_plan(config, plan_dir / "cache.yaml")
    return launch("cache_trajectory.py", [plan], gpus=args.gpus, dry_run=args.dry_run)


def train(args):
    config = configuration(args, "train")
    output = Path(args.output_dir or config["run"]["output_dir"]).resolve()
    config["run"]["output_dir"] = str(output)
    if args.cache_dir:
        config["data"]["caches"][0]["path"] = str(Path(args.cache_dir).resolve())
    if args.steering_onset is not None:
        config["training"]["transition"]["start_step"] = args.steering_onset
    for key, value in (
        ("max_steps", args.max_updates),
        ("learning_rate", args.learning_rate),
    ):
        if value is not None:
            config["training"][key] = value
    plan = save_plan(config, output.parent / ".plans" / output.name / "train.yaml")
    command = [plan]
    if args.resume:
        command += ["--resume", args.resume]
    if args.stop_at_update:
        command += ["--stop-at-step", args.stop_at_update]
    code = launch("train.py", command, gpus=args.gpus, dry_run=args.dry_run)
    if code or args.dry_run or args.stop_at_update or not args.inference_suite:
        return code
    args.run_dir = str(output)
    args.step = config["training"]["max_steps"]
    args.output_dir = str(output / "inference" / f"step-{args.step:08d}")
    args.checkpoint = None
    args.resume = True
    return infer(args)


def infer(args):
    if args.run_dir:
        run = Path(args.run_dir).resolve()
        config = load_config(run / "config.yaml")
        step = args.step or config["training"]["max_steps"]
        output = Path(
            args.output_dir or run / "inference" / f"step-{step:08d}"
        ).resolve()
    else:
        config = configuration(args, "train")
        output = Path(args.output_dir or ROOT / "outputs" / args.backbone).resolve()
        run = output.parent / ".plans" / output.name / "model"
        step = 2000
        if args.checkpoint:
            from .models.checkpoint import checkpoint_manifest

            checkpoint = Path(args.checkpoint).expanduser()
            if not checkpoint.is_dir():
                from huggingface_hub import snapshot_download
                from huggingface_hub.utils import validate_repo_id

                validate_repo_id(args.checkpoint)
                checkpoint = Path(
                    snapshot_download(
                        args.checkpoint,
                        revision=args.checkpoint_revision or "main",
                        local_dir=model_root() / args.checkpoint.split("/")[-1],
                    )
                )
            checkpoint = checkpoint.resolve()
            manifest = checkpoint_manifest(checkpoint)
            if (
                args.policies == ["tailsteer"]
                and not args.model_path
                and (checkpoint / "tailrefine_release.json").is_file()
            ):
                config["model"]["path"] = str(checkpoint)
            step = int(manifest["step"])
            config["inference_checkpoint"] = str(checkpoint)
        elif any(p != "base" for p in args.policies):
            raise ValueError(
                "TailSteer/transition inference requires --checkpoint or --run-dir; use --policies base for the base model"
            )
        save_plan(config, run / "config.yaml")
    suites = []
    for suite in args.inference_suite or ["demos"]:
        suites.extend(SUITE_GROUPS.get(suite, [suite]))
    failures = []
    for suite in dict.fromkeys(suites):
        try:
            from .evaluation.suites import evaluation_suite

            profile = evaluation_suite(suite)
            if suite in BENCHMARKS or profile:
                command = [
                    "--run-dir",
                    run,
                    "--step",
                    step,
                    "--bench",
                    suite,
                    "--eval-root",
                    args.eval_root,
                    "--gpus",
                    args.gpus,
                    "--stage",
                    args.stage,
                ]
                if profile:
                    command[4:6] = ["--suite", profile]
                    command += [
                        "--continue-on-error",
                        "--status-file",
                        output / f"{profile.stem}-status.json",
                    ]
                if args.limit:
                    command += ["--limit", args.limit]
                if args.judge_config:
                    command += ["--judge-config", args.judge_config]
                script = "evaluate_sequence.py" if profile else "evaluate_experiment.py"
            else:
                # Kontext supports a single reference image. Multi-reference demos
                # are supported by FLUX.2, Qwen and SenseNova.
                if (
                    config["model"]["name"] == "flux1_kontext_dev"
                    and suite in SUITE_GROUPS["demos"][1:]
                ):
                    print(f"Skipping {suite}: Kontext supports one reference image.")
                    continue
                command = [
                    "--run-dir",
                    run,
                    "--step",
                    step,
                    "--suite",
                    suite,
                    "--gpus",
                    args.gpus,
                    "--output-dir",
                    output / Path(suite).stem,
                ]
                script = "inference_experiment.py"
            for policy in args.policies:
                command += ["--policy", policy]
            if args.resume:
                command.append("--resume")
            if args.dry_run:
                command.append("--dry-run")
            code = launch(script, command)
        except (ValueError, OSError) as error:
            print(f"{suite}: {error}", file=sys.stderr)
            code = 1
        if code:
            failures.append(suite)
        output.mkdir(parents=True, exist_ok=True)
        (output / "suite-status.json").write_text(
            json.dumps({"last_suite": suite, "failed": failures}, indent=2)
        )
    return int(bool(failures))


def main(argv=None):
    p = argparse.ArgumentParser(prog="tailsteer", description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="Download source data or benchmark assets"
    )
    prep.add_argument(
        "dataset", choices=("model", "training-source", "add-preserve", *BENCHMARKS)
    )
    prep.add_argument("--dataset-repo")
    prep.add_argument("--backbone", choices=BACKBONES, default="flux2-klein-base-9b")
    prep.add_argument("--revision")
    prep.add_argument("--output-dir")
    prep.add_argument("--eval-root", default=str(data_root() / "evaluation"))
    prep.set_defaults(handler=prepare)
    for name, handler in (("cache", cache), ("train", train), ("infer", infer)):
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--backbone", choices=BACKBONES, default="flux2-klein-base-9b")
        sub.add_argument("--config", type=Path)
        sub.add_argument("--model-path")
        sub.add_argument("--model-code")
        sub.add_argument("--output-dir")
        sub.add_argument("--gpus", type=int, default=8)
        sub.add_argument("--dry-run", action="store_true")
        sub.add_argument("--limit", type=int)
        if name == "cache":
            sub.add_argument("--sources")
            sub.add_argument("--image-root")
            sub.add_argument(
                "--disable-filter", action="store_true", help="Only for smoke tests"
            )
            continue
        sub.add_argument("--inference-suite", action="append")
        sub.add_argument(
            "--policies",
            nargs="+",
            choices=("base", "tailsteer", "transition"),
            default=["base", "tailsteer"],
        )
        sub.add_argument("--eval-root", default=str(data_root() / "evaluation"))
        sub.add_argument("--stage", choices=("generate", "score", "all"), default="all")
        sub.add_argument("--judge-config")
        if name == "train":
            sub.add_argument("--cache-dir")
            sub.add_argument("--steering-onset", type=int)
            sub.add_argument("--max-updates", type=int)
            sub.add_argument("--learning-rate", type=float)
            sub.add_argument("--resume", nargs="?", const="auto")
            sub.add_argument("--stop-at-update", type=int)
        else:
            sub.add_argument(
                "--checkpoint",
                help="Inference export, native TailSteer directory, or Hugging Face model ID",
            )
            sub.add_argument(
                "--checkpoint-revision", help="Hugging Face model revision"
            )
            sub.add_argument("--run-dir")
            sub.add_argument("--step", type=int)
            sub.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if getattr(args, "gpus", 1) < 1:
        p.error("--gpus must be positive")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
