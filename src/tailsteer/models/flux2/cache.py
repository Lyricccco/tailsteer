"""FLUX.2 cache image preparation and preservation-mask semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from tailsteer.models.masks import dilated_keep_mask, project_keep_mask


def fit_rgb(path: str | Path, width: int, height: int) -> Image.Image:
    with Image.open(Path(path).expanduser()) as image:
        return ImageOps.fit(
            image.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
        )


def _rgb(image: Image.Image, size: int = 512) -> np.ndarray:
    return np.asarray(
        image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _perceptual_difference(
    source: Image.Image, target: Image.Image, size: int = 512
) -> tuple[np.ndarray, np.ndarray, float]:
    source_u8, target_u8 = _rgb(source, size), _rgb(target, size)
    source_f = source_u8.astype(np.float32) / 255.0
    target_f = target_u8.astype(np.float32) / 255.0
    source_mean = cv2.GaussianBlur(source_f, (11, 11), 1.5)
    target_mean = cv2.GaussianBlur(target_f, (11, 11), 1.5)
    source_variance = (
        cv2.GaussianBlur(source_f * source_f, (11, 11), 1.5) - source_mean**2
    )
    target_variance = (
        cv2.GaussianBlur(target_f * target_f, (11, 11), 1.5) - target_mean**2
    )
    covariance = (
        cv2.GaussianBlur(source_f * target_f, (11, 11), 1.5) - source_mean * target_mean
    )
    ssim_map = (
        (2 * source_mean * target_mean + 0.01**2) * (2 * covariance + 0.03**2)
    ) / np.maximum(
        (source_mean**2 + target_mean**2 + 0.01**2)
        * (source_variance + target_variance + 0.03**2),
        1e-12,
    )
    spatial_ssim = ssim_map.mean(axis=2)
    source_blur = cv2.GaussianBlur(source_f, (5, 5), 0)
    target_blur = cv2.GaussianBlur(target_f, (5, 5), 0)
    rgb_mae = np.abs(source_blur - target_blur).mean(axis=2)
    difference = cv2.GaussianBlur(
        0.55 * rgb_mae + 0.45 * np.clip(1.0 - spatial_ssim, 0.0, 1.0),
        (0, 0),
        sigmaX=1.5,
    )
    return (
        difference.astype(np.float32),
        rgb_mae.astype(np.float32),
        float(spatial_ssim.mean()),
    )


def _changed_components(
    binary: np.ndarray, *, policy: str
) -> tuple[np.ndarray, float, float]:
    if policy not in {"largest", "all"}:
        raise ValueError("cache.filter.component_policy must be largest or all")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum = max(16, int(round(binary.size * 0.0005)))
    components = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum
    ]
    if not components:
        return np.zeros_like(binary), 0.0, 0.0
    components.sort(reverse=True)
    area, label = components[0]
    dominance = area / max(sum(value for value, _ in components), 1)
    selected = components[:1] if policy == "largest" else components
    component = np.uint8(np.isin(labels, [value[1] for value in selected]))
    # The union's box is a locality metric only; never fill the space between
    # disconnected changes in the actual mask.
    rows, columns = np.nonzero(component)
    box = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
    return component, float(dominance), float(box / binary.size)


def largest_true_square(mask: np.ndarray, multiple: int = 8) -> tuple[int, int, int]:
    """Return ``(top, left, side)`` for the largest aligned all-keep square."""

    if mask.ndim != 2:
        raise ValueError(f"keep mask must be 2D, got {mask.shape}")
    if multiple < 1:
        raise ValueError("patch multiple must be positive")
    height, width = mask.shape
    dynamic = np.zeros((height + 1, width + 1), dtype=np.int32)
    best_side = best_bottom = best_right = 0
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            if mask[row - 1, column - 1]:
                side = 1 + min(
                    dynamic[row - 1, column],
                    dynamic[row, column - 1],
                    dynamic[row - 1, column - 1],
                )
                dynamic[row, column] = side
                if side > best_side:
                    best_side, best_bottom, best_right = int(side), row, column
    rounded = best_side - best_side % multiple
    if rounded == 0:
        return 0, 0, 0
    original_top = best_bottom - best_side
    original_left = best_right - best_side
    inset = (best_side - rounded) // 2
    return int(original_top + inset), int(original_left + inset), int(rounded)


def local_edit_mask(
    source: Image.Image,
    target: Image.Image,
    *,
    latent_height: int,
    latent_width: int,
    dilation_pixels: int = 48,
    patch_multiple: int = 8,
    component_policy: str = "largest",
) -> dict[str, Any]:
    """Exclude detected changes; default to legacy largest-component selection."""

    difference, _, ssim = _perceptual_difference(source, target)
    difference_u8 = np.uint8(np.rint(np.clip(difference, 0, 1) * 255))
    otsu, binary = cv2.threshold(
        difference_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component, dominance, box_ratio = _changed_components(
        np.uint8(binary > 0), policy=component_policy
    )
    dilation = max(1, round(dilation_pixels * 512 / max(source.size)))
    edit_small = cv2.dilate(
        component,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2),
    )
    edit = cv2.resize(edit_small, source.size, interpolation=cv2.INTER_NEAREST).astype(
        bool
    )
    keep_mask = project_keep_mask(edit, latent_height, latent_width)
    patch_top, patch_left, patch_side = largest_true_square(
        keep_mask, multiple=patch_multiple
    )
    return {
        "keep_mask": keep_mask,
        "edit_mask": edit,
        "change_mask": cv2.resize(
            component, source.size, interpolation=cv2.INTER_NEAREST
        ).astype(bool),
        "metrics": {
            "ssim": ssim,
            "otsu_threshold": float(otsu / 255.0),
            "raw_edit_ratio": float(component.mean()),
            "edit_ratio_after_dilation": float(edit.mean()),
            "dominant_component_ratio": dominance,
            "bounding_box_ratio": box_ratio,
            "latent_keep_ratio": float(keep_mask.mean()),
            "patch_top": patch_top,
            "patch_left": patch_left,
            "patch_side": patch_side,
        },
    }


def flux2_keep_mask_variants(
    change_mask: np.ndarray,
    *,
    latent_shape: tuple[int, int, int],
    variants: list[str],
) -> dict[str, np.ndarray]:
    """Additional masks only; neither the legacy mask nor acceptance is changed.

    Fine maps to the pre-patchify VAE grid and folds each 2x2 sub-position into
    channels in exactly FLUX.2's (feature, dy, dx, h, w) order. Dilate5 expands
    the same cleaned change mask by five IMAGE pixels before token-grid pooling.
    """
    channels, height, width = latent_shape
    if channels % 4 or set(variants) - {"fine", "dilate5"}:
        raise ValueError("invalid FLUX.2 mask variants or latent channel count")
    result = {}
    if "fine" in variants:
        keep = project_keep_mask(change_mask, height * 2, width * 2)
        features = np.broadcast_to(keep, (channels // 4, height * 2, width * 2))
        result["fine"] = (
            features.reshape(channels // 4, height, 2, width, 2)
            .transpose(0, 2, 4, 1, 3)
            .reshape(channels, height, width)
            .astype(np.uint8)
        )
    if "dilate5" in variants:
        result["dilate5"] = dilated_keep_mask(change_mask, height, width, radius=5)
    return result


def cache_mask_decision(
    *,
    source: Image.Image,
    generated: Image.Image,
    edited: Image.Image | None,
    latent_height: int,
    latent_width: int,
    filter_config: dict[str, Any],
    recipe: str | None = None,
) -> dict[str, Any]:
    """Filter a trajectory and mask its actual changes, not reference changes.

    The builder always supplies ``recipe``. Omission supports old paired/no-op
    callers only; reference-free local edits must identify their task explicitly.
    """

    reference_mode = str(filter_config.get("reference_mode", "paired"))
    if reference_mode not in {"paired", "none"}:
        raise ValueError("cache.filter.reference_mode must be paired or none")
    component_policy = str(filter_config.get("component_policy", "largest"))
    if component_policy not in {"largest", "all"}:
        raise ValueError("cache.filter.component_policy must be largest or all")
    if recipe is None:
        if reference_mode == "none":
            raise ValueError("reference_mode=none requires an explicit cache recipe")
        recipe = "noop-filtered" if edited is None else "local-edit-filtered"
    if recipe not in {"local-edit-filtered", "noop-filtered"}:
        raise ValueError(f"unsupported cache recipe: {recipe!r}")

    if recipe == "noop-filtered":
        difference, rgb, ssim = _perceptual_difference(source, generated)
        threshold = float(filter_config.get("change_threshold", 0.22))
        changed = difference >= threshold
        metrics = {
            "ssim": ssim,
            "rgb_mae": float(rgb.mean()),
            "rgb_p95": float(np.percentile(rgb, 95)),
            "changed_ratio": float(changed.mean()),
            "change_threshold": threshold,
        }
        accepted = (
            ssim >= float(filter_config.get("min_noop_ssim", 0.82))
            and metrics["rgb_mae"]
            <= float(filter_config.get("max_noop_rgb_mae", 0.065))
            and metrics["changed_ratio"]
            <= float(filter_config.get("max_changed_ratio", 0.10))
        )
        return {
            "accepted": accepted,
            "keep_mask": np.ones((latent_height, latent_width), dtype=bool),
            "metrics": metrics,
        }

    if reference_mode == "paired" and edited is None:
        raise ValueError("paired local-edit filtering requires an edited image")
    actual = local_edit_mask(
        source,
        generated,
        latent_height=latent_height,
        latent_width=latent_width,
        dilation_pixels=int(filter_config.get("dilation_pixels", 48)),
        patch_multiple=int(filter_config.get("patch_multiple", 8)),
        component_policy=component_policy,
    )
    metrics = actual["metrics"]
    accepted = (
        metrics["ssim"] >= float(filter_config.get("min_ssim", 0.72))
        and float(filter_config.get("min_raw_edit_ratio", 0.002))
        <= metrics["raw_edit_ratio"]
        <= float(filter_config.get("max_raw_edit_ratio", 0.20))
        and metrics["dominant_component_ratio"]
        >= float(filter_config.get("min_component_dominance", 0.65))
        and metrics["bounding_box_ratio"]
        <= float(filter_config.get("max_bounding_box_ratio", 0.35))
        and metrics["patch_side"] >= int(filter_config.get("min_patch_side", 16))
    )
    filter_metrics = {"generated_mask": metrics}
    if reference_mode == "paired":
        intended = local_edit_mask(
            source,
            edited,
            latent_height=latent_height,
            latent_width=latent_width,
            dilation_pixels=int(filter_config.get("dilation_pixels", 48)),
            patch_multiple=int(filter_config.get("patch_multiple", 8)),
            component_policy=component_policy,
        )
        difference, rgb, _ = _perceptual_difference(source, generated)
        intended_small = cv2.resize(
            np.uint8(intended["edit_mask"]), (512, 512), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        threshold = float(filter_config.get("generated_change_threshold", 0.22))
        changed = difference >= threshold
        outside, inside = ~intended_small, intended_small
        locality = {
            "outside_changed_ratio": float(
                changed[outside].mean() if outside.any() else 0
            ),
            "outside_rgb_mae": float(rgb[outside].mean() if outside.any() else 0),
            "inside_changed_ratio": float(
                changed[inside].mean() if inside.any() else 0
            ),
        }
        accepted = (
            accepted
            and locality["outside_changed_ratio"]
            <= float(filter_config.get("max_outside_changed_ratio", 0.10))
            and locality["outside_rgb_mae"]
            <= float(filter_config.get("max_outside_rgb_mae", 0.08))
            and locality["inside_changed_ratio"]
            >= float(filter_config.get("min_inside_changed_ratio", 0.01))
        )
        filter_metrics["generated_locality"] = locality
    return {
        "accepted": accepted,
        "keep_mask": actual["keep_mask"],
        "change_mask": actual["change_mask"],
        "metrics": filter_metrics,
    }
