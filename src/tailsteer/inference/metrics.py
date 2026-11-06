"""Useful latent and image comparisons against the pretrained policy."""

from __future__ import annotations

import math
from typing import Any


def latent_comparison_metrics(
    candidate_latent: Any, baseline_latent: Any
) -> dict[str, float]:
    if baseline_latent is None:
        return {}  # A single-policy run has no baseline, not zero error.
    import torch

    candidate = candidate_latent.detach().float()
    baseline = baseline_latent.detach().float()
    difference = candidate - baseline
    return {
        "latent_mse": float(difference.square().mean()),
        "latent_mae": float(difference.abs().mean()),
        "latent_cosine": float(
            torch.nn.functional.cosine_similarity(
                candidate.flatten(1), baseline.flatten(1), dim=1
            ).mean()
        ),
    }


def comparison_metrics(
    candidate_latent: Any,
    baseline_latent: Any,
    candidate_image: Any,
    baseline_image: Any,
) -> dict[str, float]:
    if baseline_latent is None or baseline_image is None:
        return {}
    import numpy as np

    candidate_pixels = (
        np.asarray(candidate_image.convert("RGB"), dtype=np.float32) / 255
    )
    baseline_pixels = np.asarray(baseline_image.convert("RGB"), dtype=np.float32) / 255
    image_mse = float(np.square(candidate_pixels - baseline_pixels).mean())
    return {
        **latent_comparison_metrics(candidate_latent, baseline_latent),
        "image_mse": image_mse,
        "image_psnr": 120.0 if image_mse == 0 else -10.0 * math.log10(image_mse),
    }


def image_comparison_metrics(
    candidate_image: Any, baseline_image: Any
) -> dict[str, float]:
    """Pixel-space comparison for models without a shared latent representation."""

    if baseline_image is None:
        return {}
    import cv2
    import numpy as np

    candidate = np.asarray(candidate_image.convert("RGB"), dtype=np.float32) / 255
    baseline = np.asarray(baseline_image.convert("RGB"), dtype=np.float32) / 255
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline image sizes differ")
    difference = candidate - baseline
    mse = float(np.square(difference).mean())
    x, y = candidate, baseline
    c1, c2 = 0.01**2, 0.03**2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return {
        "image_mse": mse,
        "image_mae": float(np.abs(difference).mean()),
        "image_psnr": 120.0 if mse == 0 else -10.0 * math.log10(mse),
        "image_ssim": float(ssim.mean()),
    }
