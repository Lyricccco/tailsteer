"""Deterministic overview images and contact sheets for trajectory caches."""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PANEL_SIDE = 320
CONTACT_WIDTH = 900
CONTACT_COLUMNS = 2
CONTACT_GROUP_SIZE = 8
_SAMPLE_DIRECTORY = re.compile(r"^(?P<ordinal>[0-9]{9})_(?P<sample>.+)$")


def safe_sample_id(value: str) -> str:
    """Return the same conservative path component used by legacy caches."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized[:160] or "sample"


def sample_directory(root: str | Path, ordinal: int, source_id: str) -> Path:
    if ordinal < 0 or ordinal > 999_999_999:
        raise ValueError(f"visualization ordinal is outside nine digits: {ordinal}")
    return (
        Path(root)
        / "visualizations"
        / "samples"
        / f"{ordinal:09d}_{safe_sample_id(source_id)}"
    )


def sample_visualization_complete(
    root: str | Path, ordinal: int, source_id: str
) -> bool:
    directory = sample_directory(root, ordinal, source_id)
    return (directory / "overview.png").is_file() and (
        directory / "generated.png"
    ).is_file()


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _labeled_panel(image: Image.Image, label: str) -> Image.Image:
    panel = Image.new("RGB", (PANEL_SIDE, PANEL_SIDE + 42), "white")
    panel.paste(
        ImageOps.fit(
            image.convert("RGB"),
            (PANEL_SIDE, PANEL_SIDE),
            method=Image.Resampling.NEAREST,
        ),
        (0, 42),
    )
    ImageDraw.Draw(panel).text((8, 9), label, fill="black", font=_font(18))
    return panel


def _mask_image(mask: np.ndarray) -> Image.Image:
    if mask.ndim != 2:
        raise ValueError(f"visualization mask must be 2D, got {mask.shape}")
    return Image.fromarray(np.uint8(mask.astype(bool)) * 255).convert("RGB")


def _heatmap_image(values: np.ndarray) -> Image.Image:
    if values.ndim != 2:
        raise ValueError(f"visualization heatmap must be 2D, got {values.shape}")
    upper = max(float(np.percentile(values, 99.0)), 1e-8)
    normalized = np.uint8(np.rint(np.clip(values / upper, 0.0, 1.0) * 255.0))
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def _keep_patch_overlay(
    image: Image.Image,
    keep_mask: np.ndarray,
    patch: tuple[int, int, int],
) -> Image.Image:
    top, left, side = map(int, patch)
    if side <= 0:
        return image.convert("RGB")
    height, width = keep_mask.shape
    if top < 0 or left < 0 or top + side > height or left + side > width:
        raise ValueError(f"keep patch {patch} is outside mask {keep_mask.shape}")
    if not np.asarray(keep_mask[top : top + side, left : left + side]).all():
        raise ValueError(f"keep patch {patch} contains edited latent cells")
    result = image.convert("RGB").copy()
    scale_x, scale_y = result.width / width, result.height / height
    box = (
        int(round(left * scale_x)),
        int(round(top * scale_y)),
        int(round((left + side) * scale_x)) - 1,
        int(round((top + side) * scale_y)) - 1,
    )
    ImageDraw.Draw(result).rectangle(box, outline=(20, 255, 60), width=8)
    return result


def _state_array(value: Any, *, grid: tuple[int, int], key: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 3 and 1 in array.shape:
        array = np.squeeze(
            array, axis=next(i for i, size in enumerate(array.shape) if size == 1)
        )
    height, width = grid
    pixels = height * width
    if array.ndim == 3 and tuple(array.shape[-2:]) == grid:
        return array.transpose(1, 2, 0)
    if array.ndim == 3 and tuple(array.shape[:2]) == grid:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[0] == pixels:
        return array.reshape(height, width, array.shape[1])
    if array.ndim == 2 and array.shape[1] == pixels:
        return array.T.reshape(height, width, array.shape[0])
    if array.size % pixels == 0:
        return array.reshape(height, width, array.size // pixels)
    raise ValueError(f"state {key!r} with shape {array.shape} cannot use grid {grid}")


def _joint_state_pca(
    states: Mapping[str, Any], *, grid: tuple[int, int]
) -> dict[str, Image.Image]:
    arrays = {
        key: _state_array(value, grid=grid, key=key) for key, value in states.items()
    }
    channel_counts = {value.shape[2] for value in arrays.values()}
    if len(channel_counts) != 1:
        raise ValueError("all visualization states must have the same channel count")
    rows = []
    for value in arrays.values():
        flat = value.reshape(-1, value.shape[2])
        # Keep PCA bounded for high-resolution native pixel/token states.
        stride = max(1, math.ceil(len(flat) / 8192))
        rows.append(flat[::stride])
    matrix = np.concatenate(rows).astype(np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    count = min(3, eigenvectors.shape[0])
    components = eigenvectors[:, np.argsort(eigenvalues)[-count:][::-1]]
    for column in range(count):
        pivot = int(np.argmax(np.abs(components[:, column])))
        if components[pivot, column] < 0:
            components[:, column] *= -1
    projected = {}
    for key, value in arrays.items():
        projection = (value.reshape(-1, value.shape[2]) - mean) @ components
        projection = projection.reshape(*grid, count)
        if count < 3:
            projection = np.pad(projection, ((0, 0), (0, 0), (0, 3 - count)))
        projected[key] = projection
    combined = np.concatenate([value.reshape(-1, 3) for value in projected.values()])
    scale = np.maximum(np.percentile(np.abs(combined), 99.0, axis=0), 1e-8)
    return {
        key: Image.fromarray(
            np.uint8(np.rint(np.clip(0.5 + 0.5 * value / scale, 0.0, 1.0) * 255.0))
        )
        for key, value in projected.items()
    }


def _flatten_metrics(value: Mapping[str, Any], prefix: str = "") -> list[str]:
    result = []
    for key in sorted(value):
        item = value[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.extend(_flatten_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result.append(f"{name}={float(item):.4f}")
    return result


def make_trajectory_overview(
    *,
    source_id: str,
    source: Image.Image,
    reference: Image.Image | None,
    generated: Image.Image,
    keep_mask: np.ndarray,
    states: Mapping[str, Any],
    grid: tuple[int, int],
    prompt: str,
    metrics: Mapping[str, Any],
    change_threshold: float,
    keep_patch: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Render the same 4-column diagnostic layout used by the Flux2 cache."""

    from tailsteer.models.flux2.cache import _perceptual_difference

    difference, _, _ = _perceptual_difference(source, generated)
    changed = difference >= change_threshold
    pca = _joint_state_pca(states, grid=grid)
    target = generated if reference is None else reference
    target_label = "generated output" if reference is None else "edited reference"
    edit_overlay = source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    overlay = np.asarray(edit_overlay).copy()
    overlay[changed] = np.uint8(
        0.45 * overlay[changed] + 0.55 * np.array([255, 40, 40])
    )
    patch_panel = (
        _keep_patch_overlay(Image.fromarray(overlay), keep_mask, keep_patch)
        if keep_patch is not None and int(keep_patch[2]) > 0
        else Image.fromarray(overlay)
    )
    patch_label = (
        "generated change + keep patch"
        if keep_patch is not None and int(keep_patch[2]) > 0
        else "generated change overlay"
    )
    rows = [
        [
            _labeled_panel(source, "source"),
            _labeled_panel(target, target_label),
            _labeled_panel(generated, "generated"),
            _labeled_panel(_heatmap_image(difference), "difference heatmap"),
        ],
        [
            _labeled_panel(_mask_image(changed), "generated change mask"),
            _labeled_panel(_mask_image(keep_mask), "latent keep mask"),
            _labeled_panel(patch_panel, patch_label),
            _labeled_panel(pca["z_src"], "z_src PCA-RGB"),
        ],
    ]
    state_keys = [key for key in sorted(states) if key != "z_src"][:3]
    third = [_labeled_panel(pca[key], f"{key} PCA-RGB") for key in state_keys]
    while len(third) < 4:
        third.append(_labeled_panel(Image.new("RGB", (64, 64), "white"), ""))
    rows.append(third)
    prompt_lines = textwrap.wrap(str(prompt), width=135) or [""]
    footer_height = 100 + 24 * len(prompt_lines)
    panel_width, row_height = rows[0][0].size
    canvas = Image.new(
        "RGB", (panel_width * 4, row_height * 3 + footer_height), "white"
    )
    for row_index, panels in enumerate(rows):
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * panel_width, row_index * row_height))
    metric_text = " | ".join(_flatten_metrics(metrics)[:6])
    if keep_patch is not None and int(keep_patch[2]) > 0:
        top, left, side = map(int, keep_patch)
        metric_text += f" | keep_patch={side}x{side}@({top},{left})"
    draw = ImageDraw.Draw(canvas)
    y = row_height * 3 + 12
    draw.text(
        (12, y), f"sample={source_id} | {metric_text}", fill="black", font=_font(17)
    )
    y += 31
    draw.text((12, y), "Prompt:", fill="black", font=_font(18))
    y += 27
    for line in prompt_lines:
        draw.text((12, y), line, fill=(30, 30, 30), font=_font(16))
        y += 22
    return canvas


def save_sample_visualization(
    *,
    root: str | Path,
    ordinal: int,
    source_id: str,
    source: Image.Image,
    reference: Image.Image | None,
    generated: Image.Image,
    keep_mask: np.ndarray,
    states: Mapping[str, Any],
    grid: tuple[int, int],
    prompt: str,
    metrics: Mapping[str, Any],
    change_threshold: float = 0.22,
    keep_patch: tuple[int, int, int] | None = None,
) -> Path:
    directory = sample_directory(root, ordinal, source_id)
    overview = make_trajectory_overview(
        source_id=source_id,
        source=source,
        reference=reference,
        generated=generated,
        keep_mask=keep_mask,
        states=states,
        grid=grid,
        prompt=prompt,
        metrics=metrics,
        change_threshold=change_threshold,
        keep_patch=keep_patch,
    )
    # Visuals are intentionally published before the tensor record. A process
    # crash can therefore leave an overwrite-safe orphan, never a committed
    # tensor whose required overview is missing.
    _atomic_save_png(overview, directory / "overview.png")
    _atomic_save_png(generated.convert("RGB"), directory / "generated.png")
    return directory


def build_contact_sheet(paths: Sequence[str | Path], output: str | Path) -> None:
    if not paths:
        return
    thumbnails = []
    for path in paths:
        with Image.open(path) as image:
            height = max(1, round(image.height * CONTACT_WIDTH / image.width))
            thumbnails.append(
                image.convert("RGB").resize(
                    (CONTACT_WIDTH, height), Image.Resampling.BICUBIC
                )
            )
    rows = math.ceil(len(thumbnails) / CONTACT_COLUMNS)
    row_heights = [
        max(
            thumbnail.height
            for thumbnail in thumbnails[
                row * CONTACT_COLUMNS : (row + 1) * CONTACT_COLUMNS
            ]
        )
        for row in range(rows)
    ]
    canvas = Image.new(
        "RGB", (CONTACT_WIDTH * CONTACT_COLUMNS, sum(row_heights)), (225, 225, 225)
    )
    y = 0
    for row, height in enumerate(row_heights):
        start = row * CONTACT_COLUMNS
        for column, thumbnail in enumerate(thumbnails[start : start + CONTACT_COLUMNS]):
            canvas.paste(thumbnail, (column * CONTACT_WIDTH, y))
        y += height
    _atomic_save_png(canvas, Path(output))


def build_visualization_contact_sheets(
    root: str | Path, *, group_size: int = CONTACT_GROUP_SIZE
) -> dict[str, int]:
    root = Path(root)
    overviews = sorted((root / "visualizations" / "samples").glob("*/overview.png"))
    contact_root = root / "visualizations" / "contact_sheets"
    expected = set()
    for start in range(0, len(overviews), group_size):
        group = overviews[start : start + group_size]
        first = group[0].parent.name.split("_", 1)[0]
        last = group[-1].parent.name.split("_", 1)[0]
        output = contact_root / f"contact_sheet_{first}_{last}.png"
        expected.add(output)
        build_contact_sheet(group, output)
    if contact_root.exists():
        for stale in contact_root.glob("contact_sheet_*.png"):
            if stale not in expected:
                stale.unlink()
    return {
        "sample_overviews": len(overviews),
        "contact_sheets": math.ceil(len(overviews) / group_size),
    }


def visualization_sample_directories(root: str | Path) -> tuple[tuple[int, Path], ...]:
    samples = Path(root) / "visualizations" / "samples"
    result = []
    seen = set()
    for directory in sorted(samples.glob("*")):
        if not directory.is_dir():
            continue
        match = _SAMPLE_DIRECTORY.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"invalid visualization sample directory: {directory}")
        ordinal = int(match.group("ordinal"))
        if ordinal in seen:
            raise ValueError(f"duplicate visualization ordinal in {root}: {ordinal}")
        seen.add(ordinal)
        for required in ("overview.png", "generated.png"):
            if not (directory / required).is_file():
                raise FileNotFoundError(directory / required)
        result.append((ordinal, directory))
    return tuple(result)


def validate_visualizations(
    root: str | Path,
    summary: Mapping[str, Any],
    *,
    expected_samples: int | None = None,
) -> dict[str, int]:
    directories = visualization_sample_directories(root)
    contact_sheets = tuple(
        sorted(
            (Path(root) / "visualizations" / "contact_sheets").glob(
                "contact_sheet_*.png"
            )
        )
    )
    actual = {
        "sample_overviews": len(directories),
        "contact_sheets": len(contact_sheets),
    }
    expected = {key: int(summary[key]) for key in actual}
    if actual != expected:
        raise ValueError(
            f"visualization summary differs: actual={actual}, expected={expected}"
        )
    if expected_samples is not None and actual["sample_overviews"] != expected_samples:
        raise ValueError(
            "visualization/source count differs: "
            f"{actual['sample_overviews']} != {expected_samples}"
        )
    if actual["contact_sheets"] != math.ceil(
        actual["sample_overviews"] / CONTACT_GROUP_SIZE
    ):
        raise ValueError("visualization contact-sheet count is not canonical")
    return actual


def write_visualization_manifest(root: str | Path, summary: Mapping[str, Any]) -> None:
    target = Path(root) / "visualizations" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, target)
