"""Build a portable, reproducible inference plan."""

from __future__ import annotations
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from tailsteer.utils import load_config
from .backends import InferenceBuildContext
from .registry import inference_backend_for_model

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_NAMES = ("base", "tailsteer")
FINAL_POLICY_NAMES = ("base", "transition", "tailsteer")
ALL_POLICY_NAMES = FINAL_POLICY_NAMES
_SUITE_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,100}[a-z0-9])?\Z")


@dataclass(frozen=True)
class PreparedInference:
    backend: str
    model_name: str
    training_run: Path
    export_path: Path
    export_step: int
    suite_name: str
    output_dir: Path
    config_path: Path
    plan_path: Path
    config: dict[str, Any]
    plan: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        policy_config = self.plan.get("policies")
        if policy_config is None:
            policy_config = self.plan.get("execution", {}).get("plans", {})
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "training_run": str(self.training_run),
            "export_path": str(self.export_path),
            "export_step": self.export_step,
            "suite": self.suite_name,
            "policies": list(policy_config),
            "output_dir": str(self.output_dir),
            "config_path": str(self.config_path),
            "plan_path": str(self.plan_path),
        }


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"YAML must contain a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def resolve_suite(value: str | Path) -> tuple[Path, dict[str, Any]]:
    raw = Path(value).expanduser()
    if raw.is_absolute() or raw.parent != Path(".") or raw.suffix:
        path = raw.resolve()
    else:
        filename = str(value).replace("-", "_") + ".yaml"
        path = (REPOSITORY / "configs/inference/suites" / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"inference suite does not exist: {path}")
    suite = load_config(path)
    if int(suite.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported inference suite schema: {path}")
    name = str(suite.get("name", ""))
    if not _SUITE_NAME.fullmatch(name):
        raise ValueError(f"invalid inference suite name {name!r}")
    if int(suite.get("num_steps", 0)) < 1:
        raise ValueError("inference suite num_steps must be positive")
    task_type = suite.get("task_type", "editing")
    if task_type not in {"editing", "t2i"}:
        raise ValueError(f"unsupported inference task_type {task_type!r}")
    required = ("tasks",) if task_type == "t2i" else ("tasks", "image_root")
    for field in required:
        if field not in suite:
            raise KeyError(f"inference suite requires {field}")
    tasks = Path(str(suite["tasks"])).expanduser().resolve()
    if not tasks.is_file():
        raise FileNotFoundError(f"inference suite task file does not exist: {tasks}")
    suite["tasks"] = str(tasks)
    if task_type == "editing":
        image_root = Path(str(suite["image_root"])).expanduser().resolve()
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"inference suite image root does not exist: {image_root}"
            )
        suite["image_root"] = str(image_root)
    return path, suite


def _segments(start: int, end: int) -> list[dict[str, Any]]:
    if not 0 <= start < end:
        raise ValueError(f"invalid refined actor interval [{start}, {end})")
    if start == 0:
        return [{"start": 0, "end": end, "actor": "refined"}]
    return [
        {"start": 0, "end": start, "actor": "base"},
        {"start": start, "end": end, "actor": "refined"},
    ]


def build_final_policies(
    *,
    num_steps: int,
    transition_actor_index: int,
    selected: Sequence[str] | None = None,
) -> dict[str, Any]:
    """All final recipes use native steps and one ordered three-way comparison."""
    names = tuple(DEFAULT_POLICY_NAMES if selected is None else selected)
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - set(FINAL_POLICY_NAMES)
    ):
        raise ValueError(
            f"final inference policies must select from {FINAL_POLICY_NAMES}"
        )
    if not 0 <= transition_actor_index < num_steps:
        raise ValueError("tail start is outside the native forward schedule")
    starts = {
        "base": None,
        "transition": transition_actor_index,
        "tailsteer": 0,
    }
    return {
        name: {
            "segments": (
                [{"start": 0, "end": num_steps, "actor": "base"}]
                if starts[name] is None
                else _segments(starts[name], num_steps)
            )
        }
        for name in FINAL_POLICY_NAMES
        if name in names
    }


def _validate_export(
    export_path: Path, *, step: int, model_name: str
) -> dict[str, Any]:
    from ..models.checkpoint import checkpoint_manifest

    manifest = checkpoint_manifest(export_path)
    if int(manifest.get("step", -1)) != int(step):
        raise ValueError(
            f"export manifest step differs from requested step: "
            f"{manifest.get('step')} != {step}"
        )
    if str(manifest.get("model", "")) != model_name:
        raise ValueError(
            f"export model differs from training config: "
            f"{manifest.get('model')!r} != {model_name!r}"
        )
    weights = export_path / str(manifest.get("weights", ""))
    if not weights.is_file():
        raise FileNotFoundError(f"export weights do not exist: {weights}")
    return manifest


def prepare_inference(
    training_run,
    *,
    step,
    suite="demos",
    policies=None,
    endpoint_modes=None,
    inference_overrides=None,
    variant=None,
    output_dir=None,
):
    if endpoint_modes:
        raise ValueError("Use base, tailsteer, or transition policies")
    run = Path(training_run).expanduser().resolve()
    training_config_path = run / "config.yaml"
    training_config = load_config(training_config_path)
    model_name = training_config["model"]["name"]
    backend = inference_backend_for_model(model_name)
    suite_path, suite_config = resolve_suite(suite)
    backend_options = suite_config.setdefault("backends", {}).setdefault(
        backend.key, {}
    )
    num_steps = int(
        backend_options.get("num_steps", getattr(backend, "default_num_steps", 50))
    )
    suite_config["num_steps"] = num_steps
    backend_options["num_steps"] = num_steps
    actor_steps = backend.actor_step_count(num_steps)
    transition = training_config["training"]["transition"]
    actor_start = backend.forward_step_to_actor_index(transition["start_step"])
    selected = policies or suite_config.get("policies") or DEFAULT_POLICY_NAMES
    policy_config = build_final_policies(
        num_steps=actor_steps, transition_actor_index=actor_start, selected=selected
    )
    export_path = Path(
        training_config.get("inference_checkpoint")
        or run / "exports" / f"step-{int(step):08d}"
    )
    needs_weights = any(p != "base" for p in policy_config)
    manifest = (
        _validate_export(export_path, step=step, model_name=model_name)
        if needs_weights
        else {}
    )
    name = suite_config["name"] + (f"-{variant}" if variant else "")
    output = (
        Path(output_dir).resolve()
        if output_dir
        else run / "inference" / f"step-{step:08d}" / name
    )
    evaluation = dict(
        schema_version=1,
        backend=backend.key,
        suite=suite_config["name"],
        suite_path=str(suite_path),
        export_step=step,
        num_steps=num_steps,
        actor_steps=actor_steps,
        transition=dict(
            start_step=transition["start_step"],
            supervision_step=transition["supervision_step"],
            actor_start_index=actor_start,
        ),
    )
    context = InferenceBuildContext(
        training_run=run,
        training_config_path=training_config_path,
        training_config=training_config,
        export_path=export_path,
        export_manifest=manifest,
        export_step=step,
        suite_path=suite_path,
        suite=suite_config,
        output_dir=output,
        policies=policy_config,
        inference_overrides=dict(inference_overrides or {}),
        evaluation=evaluation,
    )
    config = backend.build_config(context)
    if not needs_weights:
        config["inference"].pop("export_path", None)
    if "add_preserve" in suite_config:
        from tailsteer.evaluation.add_preserve import configure

        configure(config, suite_config["add_preserve"])
    serialized = yaml.safe_dump(config, sort_keys=False)
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()[:12]
    plan_dir = output.parent / ".plans" / output.name
    config_path = plan_dir / f"config-{fingerprint}.yaml"
    plan_path = plan_dir / f"plan-{fingerprint}.json"
    plan = dict(
        backend=backend.key,
        model_name=model_name,
        export_path=str(export_path),
        export_step=step,
        policies=policy_config,
        suite=name,
        output_dir=str(output),
        config_path=str(config_path),
        config_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
    )
    _atomic_text(config_path, serialized)
    _atomic_text(plan_path, json.dumps(plan, indent=2) + "\n")
    return PreparedInference(
        backend.key,
        model_name,
        run,
        export_path,
        step,
        name,
        output,
        config_path,
        plan_path,
        config,
        plan,
    )
