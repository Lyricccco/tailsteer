"""Shared comparison-panel rendering for inference outputs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

COMPARISON_LAYOUT = "separate_source_and_round_outputs_prompt_v1"
# Semantic order for comparison columns.  Endpoint policies use the same
# ordering (``transition`` occupies the transition slot); unknown/custom
# policy names retain their configured order after these names.
_CANONICAL_POLICY_ORDER = (
    "base",
    "last_step_refined",
    "legacy_transition",
    "transition_refined",
    "transition",
    "tailsteer",
)


def _single_row_panel(
    images: Mapping[str, Any],
    policy_names: Sequence[str],
    *,
    label_suffix: str,
    prompt: str,
) -> Any:
    from PIL import Image, ImageDraw

    names = _stable_policy_order(tuple(str(name) for name in policy_names))
    if not names:
        raise ValueError("a comparison panel requires at least one policy")
    if len(set(names)) != len(names):
        raise ValueError("comparison panel policy names must be unique")
    missing = [name for name in names if name not in images]
    if missing:
        raise KeyError(f"comparison panel is missing images: {missing}")

    prepared = {name: images[name].convert("RGB") for name in names}
    width, height = prepared[names[0]].size
    if any(image.size != (width, height) for image in prepared.values()):
        raise ValueError("comparison panel images must share one image size")

    header = 30
    prompt_footer = 34
    prompt_line = "Prompt: " + " ".join(str(prompt).splitlines())
    row_width = width * len(names)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    prompt_box = probe.textbbox((0, 0), prompt_line)
    prompt_width = prompt_box[2] - prompt_box[0] + 12
    canvas_width = max(row_width, prompt_width)
    canvas = Image.new(
        "RGB",
        (canvas_width, height + header + prompt_footer),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(names):
        left = column * width
        draw.text((left + 6, 8), f"{name} {label_suffix}", fill="black")
        canvas.paste(prepared[name], (left, header))
    draw.text((6, header + height + 9), prompt_line, fill="black")
    return canvas


def source_comparison_panel(source: Any, policy_names: Sequence[str]) -> Any:
    """Render the same initial source once for every policy column."""

    names = tuple(str(name) for name in policy_names)
    return _single_row_panel(
        {name: source for name in names},
        names,
        label_suffix="source",
        prompt="source image",
    )


def output_comparison_panel(
    outputs: Mapping[str, Any],
    policy_names: Sequence[str],
    *,
    prompt: str,
) -> Any:
    """Render one round's policy outputs in a single horizontal row."""

    return _single_row_panel(
        outputs,
        policy_names,
        label_suffix="output",
        prompt=prompt,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _policy_order(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    # ``policies`` is serialized with sorted keys in some older outputs.  The
    # result records, however, are a list and retain the execution/config
    # order, so prefer that order when rebuilding an existing output.
    tasks = manifest.get("tasks")
    if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            records = task.get("results")
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
                continue
            names: list[str] = []
            for record in records:
                if not isinstance(record, Mapping) or "policy" not in record:
                    continue
                name = str(record["policy"])
                if name not in names:
                    names.append(name)
            if names:
                return _stable_policy_order(names)

    explicit = manifest.get("policy_order")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        names = tuple(str(name) for name in explicit)
        if names and len(set(names)) == len(names):
            return _stable_policy_order(names)

    policies = manifest.get("policies")
    if isinstance(policies, Mapping):
        return _stable_policy_order(tuple(str(name) for name in policies))
    return ()


def _stable_policy_order(names: Sequence[str]) -> tuple[str, ...]:
    available = set(names)
    ordered = [name for name in _CANONICAL_POLICY_ORDER if name in available]
    ordered.extend(name for name in names if name not in ordered)
    return tuple(ordered)


def _record_policy_order(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for record in records:
        if "policy" not in record:
            continue
        name = str(record["policy"])
        if name not in names:
            names.append(name)
    return tuple(names)


def _ordered_names(
    by_policy: Mapping[str, Any],
    *,
    record_order: Sequence[str],
    preferred_order: Sequence[str],
) -> tuple[str, ...]:
    names = _stable_policy_order(tuple(by_policy))
    names += tuple(
        name for name in record_order if name in by_policy and name not in names
    )
    names += tuple(
        name for name in preferred_order if name in by_policy and name not in names
    )
    names += tuple(name for name in by_policy if name not in names)
    return names


def _rebuild_task(
    task_path: Path,
    *,
    output: Path,
    preferred_order: Sequence[str],
) -> tuple[dict[str, Any], int]:
    from PIL import Image

    task = json.loads(task_path.read_text(encoding="utf-8"))
    records = task.get("results")
    if not isinstance(records, list):
        raise TypeError(f"task results must be a list: {task_path}")
    task_dir = task_path.parent
    comparison_dir = task_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    source_path = output / str(task["source"])
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    panel_count = 0
    source_panel_path = comparison_dir / "source.png"
    source_panel_written = False

    policy_records = [item for item in records if isinstance(item, dict)]
    if policy_records and all("policy" in item for item in policy_records):
        record_order = _record_policy_order(policy_records)
        rounds = sorted({int(item["round"]) for item in policy_records})
        for round_index in rounds:
            by_policy = {
                str(item["policy"]): item
                for item in policy_records
                if int(item["round"]) == round_index and item.get("image")
            }
            if not by_policy:
                continue
            names = _ordered_names(
                by_policy,
                record_order=record_order,
                preferred_order=preferred_order,
            )
            if set(names) != set(by_policy):
                raise ValueError(f"invalid policy ordering in {task_path}")
            images: dict[str, Any] = {}
            for name in names:
                with Image.open(output / str(by_policy[name]["image"])) as opened:
                    images[name] = opened.convert("RGB")
            if not source_panel_written:
                source_comparison_panel(source, names).save(source_panel_path)
                source_panel_written = True
            prompts = {str(item.get("prompt", "")) for item in by_policy.values()}
            if len(prompts) != 1:
                raise ValueError(f"policies used different prompts in {task_path}")
            panel_path = comparison_dir / f"round-{round_index:02d}.png"
            output_comparison_panel(
                images,
                names,
                prompt=prompts.pop(),
            ).save(panel_path)
            relative = str(panel_path.relative_to(output))
            for item in by_policy.values():
                item["comparison_panel"] = relative
            panel_count += 1
    else:
        refined_policy = str(task.get("input", {}).get("refined_policy", "refined"))
        for item in policy_records:
            if not {"round", "pretrained_image", "refined_image"} <= set(item):
                continue
            paths = {
                "base": str(item["pretrained_image"]),
                refined_policy: str(item["refined_image"]),
            }
            images = {}
            for name, relative_path in paths.items():
                with Image.open(output / relative_path) as opened:
                    images[name] = opened.convert("RGB")
            if not source_panel_written:
                source_comparison_panel(source, tuple(paths)).save(source_panel_path)
                source_panel_written = True
            panel_path = comparison_dir / f"round-{int(item['round']):02d}.png"
            output_comparison_panel(
                images,
                tuple(paths),
                prompt=str(item.get("prompt", "")),
            ).save(panel_path)
            item["comparison_panel"] = str(panel_path.relative_to(output))
            panel_count += 1

    if not source_panel_written:
        raise ValueError(f"task has no decodable comparison records: {task_path}")
    task["source_comparison"] = str(source_panel_path.relative_to(output))
    task["comparison_layout"] = COMPARISON_LAYOUT
    if policy_records and all("policy" in item for item in policy_records):
        task["policy_order"] = list(
            _stable_policy_order(_record_policy_order(policy_records))
        )
    _write_json(task_path, task)
    return task, panel_count


def rebuild_output_comparison_panels(output_dir: str | Path) -> dict[str, Any]:
    """Rewrite completed inference panels without loading a model."""

    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"inference manifest does not exist: {manifest_path}")
    if not (output / "COMPLETE").is_file():
        raise RuntimeError(f"inference output is not complete: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preferred_order = _policy_order(manifest)
    task_paths = sorted((output / "tasks").glob("*/task.json"))
    if not task_paths:
        raise FileNotFoundError(
            f"inference output contains no task manifests: {output}"
        )

    rebuilt: dict[str, dict[str, Any]] = {}
    panel_count = 0
    for task_path in task_paths:
        task, count = _rebuild_task(
            task_path,
            output=output,
            preferred_order=preferred_order,
        )
        rebuilt[str(task["id"])] = task
        panel_count += count

    manifest["tasks"] = [rebuilt[str(task["id"])] for task in manifest["tasks"]]
    manifest["comparison_layout"] = COMPARISON_LAYOUT
    if preferred_order:
        manifest["policy_order"] = list(preferred_order)
    _write_json(manifest_path, manifest)
    for shard_path in sorted(output.glob("manifest-shard-*.json")):
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        shard["tasks"] = [rebuilt[str(task["id"])] for task in shard["tasks"]]
        shard["comparison_layout"] = COMPARISON_LAYOUT
        if preferred_order:
            shard["policy_order"] = list(preferred_order)
        _write_json(shard_path, shard)
    return {
        "output_dir": str(output),
        "layout": COMPARISON_LAYOUT,
        "task_count": len(rebuilt),
        "panel_count": panel_count,
    }
