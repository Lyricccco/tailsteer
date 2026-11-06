"""Fixed-sample visual validation for full image-editing inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from tailsteer.cache.dataset import cache_record


def select_fixed_local_edit_indices(
    dataset: Any, *, count: int, seed: int
) -> tuple[int, ...]:
    candidates = []
    for index in range(len(dataset)):
        record = cache_record(dataset, index)
        if str(record["task"]) != "local_edit":
            continue
        digest = hashlib.sha256(f"{seed}:{record['sample_id']}".encode()).hexdigest()
        candidates.append((digest, index, str(record["source_identity"])))
    candidates.sort()
    selected: list[int] = []
    sources: set[str] = set()
    for _, index, source_identity in candidates:
        if source_identity in sources:
            continue
        sources.add(source_identity)
        selected.append(index)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"visual validation needs {count} source-distinct local_edit samples, "
            f"found {len(selected)}"
        )
    return tuple(selected)


def save_fixed_sample_manifest(
    dataset: Any, indices: tuple[int, ...], destination: str | Path
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for order, index in enumerate(indices):
        item = cache_record(dataset, index)
        records.append(
            {
                "order": order,
                "dataset_index": index,
                "sample_id": item["sample_id"],
                "source_identity": item["source_identity"],
                "prompt": item["prompt"],
                "seed": item["seed"],
            }
        )
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _pixel_mask(keep_mask: Any, size: tuple[int, int]) -> np.ndarray:
    value = keep_mask.detach().float().cpu().numpy()
    while value.ndim > 2:
        value = value[0]
    image = Image.fromarray((value >= 0.5).astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST)) >= 128


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def _ssim_map(source: np.ndarray, generated: np.ndarray) -> np.ndarray:
    import cv2

    x, y = source.astype(np.float32) / 255.0, generated.astype(np.float32) / 255.0
    c1, c2 = 0.01**2, 0.03**2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return value.mean(axis=2)


def image_edit_metrics(
    source: Image.Image, generated: Image.Image, keep_mask: Any
) -> dict[str, Any]:
    import cv2

    source_array = np.asarray(source.convert("RGB"), dtype=np.uint8)
    generated_array = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    if source_array.shape != generated_array.shape:
        raise ValueError("source and generated image sizes differ")
    keep = _pixel_mask(keep_mask, source.size)
    edit = ~keep
    difference = np.abs(source_array.astype(np.float32) - generated_array) / 255.0
    rgb_mae = difference.mean(axis=2)
    changed = np.linalg.norm(difference, axis=2) > 0.08
    ssim = _ssim_map(source_array, generated_array)
    gray = cv2.cvtColor(generated_array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    clipped = np.logical_or(generated_array <= 2, generated_array >= 253).mean(axis=2)
    return {
        "edit_region_change": {
            "rgb_mae": _masked_mean(rgb_mae, edit),
            "changed_pixel_ratio": _masked_mean(changed, edit),
            "pixel_fraction": float(edit.mean()),
        },
        "non_edit_region_preservation": {
            "rgb_mae": _masked_mean(rgb_mae, keep),
            "unchanged_pixel_ratio": 1.0 - _masked_mean(changed, keep),
            "ssim": _masked_mean(ssim, keep),
            "pixel_fraction": float(keep.mean()),
        },
        "full_inference_image_quality": {
            "source_relative_ssim": float(ssim.mean()),
            "sharpness_laplacian_variance": float(
                cv2.Laplacian(gray, cv2.CV_32F).var()
            ),
            "clipping_fraction": float(clipped.mean()),
            "semantic_reference_available": False,
            "visual_review_required": True,
        },
    }


def save_visual_panel(
    source: Image.Image,
    generated: Image.Image,
    keep_mask: Any,
    destination: str | Path,
) -> None:
    keep = _pixel_mask(keep_mask, source.size)
    source_array = np.asarray(source.convert("RGB"), dtype=np.uint8)
    generated_array = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    overlay = source_array.copy()
    edit = ~keep
    overlay[edit] = (0.55 * overlay[edit] + 0.45 * np.array([255, 40, 40])).astype(
        np.uint8
    )
    difference = np.abs(source_array.astype(np.int16) - generated_array).max(axis=2)
    heat = np.zeros_like(source_array)
    heat[..., 0] = np.clip(difference * 3, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(difference, 0, 255).astype(np.uint8)
    views = [source, Image.fromarray(overlay), generated, Image.fromarray(heat)]
    labels = ["source", "edit mask (red)", "generated", "absolute difference"]
    header = 28
    canvas = Image.new("RGB", (source.width * 4, source.height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (view, label) in enumerate(zip(views, labels, strict=True)):
        left = index * source.width
        canvas.paste(view, (left, header))
        draw.text((left + 6, 7), label, fill="black")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def aggregate_visual_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = (
        "edit_region_change",
        "non_edit_region_preservation",
        "full_inference_image_quality",
    )
    aggregate: dict[str, Any] = {}
    for group in groups:
        numeric_keys = {
            key
            for record in records
            for key, value in record["metrics"][group].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        aggregate[group] = {
            key: float(np.mean([record["metrics"][group][key] for record in records]))
            for key in sorted(numeric_keys)
        }
    aggregate["full_inference_image_quality"].update(
        {
            "semantic_reference_available": False,
            "visual_review_required": True,
        }
    )
    return aggregate
