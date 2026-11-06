"""Opt-in multi-mask projection, storage, and training readers (CPU only)."""

import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tailsteer.cache import trajectory
from tailsteer.cache.dataset import CacheDataset, compose_cache_datasets
from tailsteer.cache.format import (
    TRAJECTORY_GROUPED_LAYOUT,
    CacheExample,
    TrajectoryCacheWriter,
    validate_cache,
)
from tailsteer.cache.trajectory_dataset import (
    TrajectorySegmentDataset,
    TrajectorySourceDataset,
    materialize_trajectory_state,
    trajectory_source_collate,
)
from tailsteer.models.flux2.cache import flux2_keep_mask_variants, local_edit_mask
from tailsteer.models.protocols import TrajectoryPoint


@pytest.mark.parametrize("dy, dx", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_fine_matches_flux2_patchify_and_preserves_other_subpositions(dy, dx):
    change = np.zeros((32, 48), dtype=bool)
    change[dy * 8 + 1, dx * 8 + 1] = True
    fine = flux2_keep_mask_variants(
        change, latent_shape=(128, 2, 3), variants=["fine"]
    )["fine"]
    keep = ~change.reshape(4, 8, 6, 8).any(axis=(1, 3))
    vae_mask = torch.from_numpy(keep).expand(1, 32, 4, 6)
    # The same view/permute order as official Flux2Pipeline._patchify_latents.
    expected = vae_mask.reshape(1, 32, 2, 2, 3, 2).permute(0, 1, 3, 5, 2, 4)
    np.testing.assert_array_equal(fine, expected.reshape(128, 2, 3).numpy())
    # Native dev AutoEncoder.encode uses the same (feature, dy, dx) order.
    from einops import rearrange

    native = rearrange(vae_mask, "b c (h dh) (w dw) -> b (c dh dw) h w", dh=2, dw=2)
    np.testing.assert_array_equal(fine, native[0].numpy())
    assert fine.dtype == np.uint8
    assert fine[:, 0, 0].sum() == 96
    assert not fine[dy * 2 + dx :: 4, 0, 0].any()
    assert fine[:, 1, 1:].all()


def test_dilate5_means_five_original_image_pixels_not_detector_pixels():
    change = np.zeros((32, 32), dtype=bool)
    change[16, 16] = True
    # Identity projection isolates the image-space dilation before grid pooling.
    actual = flux2_keep_mask_variants(
        change, latent_shape=(128, 32, 32), variants=["dilate5"]
    )["dilate5"][0]
    expected = cv2.dilate(
        change.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    np.testing.assert_array_equal(actual, 1 - expected)
    assert actual[16, 10] and not actual[16, 11]
    assert not actual[16, 21] and actual[16, 22]


@pytest.mark.parametrize("task", ["local_edit", "noop"])
@pytest.mark.parametrize("model", ["flux2_klein_9b"])
def test_builder_adds_three_masks_without_changing_legacy_mask(
    tmp_path, monkeypatch, task, model
):
    source = Image.new("RGB", (1024, 1024), "white")
    generated = source.copy()
    ImageDraw.Draw(generated).rectangle((250, 320, 393, 447), fill="black")
    source.save(tmp_path / "source.png")
    state = torch.zeros(128, 64, 64, dtype=torch.bfloat16)
    trace = trajectory.TrajectoryResult(
        image=generated,
        source_state=state,
        forward_states={0: state, 49: state},
        final_state=state,
        schedule_points={0: (1.0, 1000.0), 49: (0.02, 20.0)},
        final_schedule_point=(0.0, 0.0),
        grid=(64, 64),
        extra={},
    )
    monkeypatch.setattr(trajectory, "_run_trajectory", lambda *_, **__: trace)
    config = {
        "model": {"name": model},
        "data": {},
        "cache": {
            "recipe": "annotated-mixed",
            "width": 1024,
            "height": 1024,
            "forward_points": [0, 49],
            "final_point": 50,
            "mask_variants": ["fine", "dilate5"],
            "filter": {
                "reference_mode": "none",
                "component_policy": "all",
                "enabled": False,
            },
        },
    }
    _, examples, _ = trajectory.prepare_trajectory_examples(
        runtime=None,
        record={"id": "a", "source": "source.png", "prompt": "edit", "task": task},
        config=config,
        image_root=tmp_path,
    )
    first = examples[0]
    assert tuple(first.keep_mask.shape) == (1, 64, 64)
    assert tuple(first.mask_variants["fine"].shape) == (128, 64, 64)
    assert tuple(first.mask_variants["dilate5"].shape) == (1, 64, 64)
    assert all(row.mask_variants is first.mask_variants for row in examples)
    if task == "noop":
        assert first.keep_mask.all()
        assert all(mask.all() for mask in first.mask_variants.values())
    else:
        old = local_edit_mask(
            source, generated, latent_height=64, latent_width=64, component_policy="all"
        )
        np.testing.assert_array_equal(first.keep_mask[0].numpy(), old["keep_mask"])
        assert (
            first.mask_variants["fine"].float().mean()
            > first.mask_variants["dilate5"].float().mean()
        )
        assert (
            first.mask_variants["dilate5"].float().mean()
            > first.keep_mask.float().mean()
        )


@pytest.fixture
def cache(tmp_path):
    change = np.zeros((32, 32), dtype=bool)
    change[2, 2] = True
    extras = {
        name: torch.from_numpy(value)
        for name, value in flux2_keep_mask_variants(
            change, latent_shape=(128, 2, 2), variants=["fine", "dilate5"]
        ).items()
    }
    examples = [
        CacheExample(
            anchor_latent=torch.full((128, 2, 2), float(step)),
            source_latent=torch.zeros(128, 2, 2),
            keep_mask=torch.zeros(1, 2, 2, dtype=torch.uint8),
            mask_variants=extras,
            metadata={
                "sample_id": f"a:{step}",
                "source_id": "a",
                "source_identity": "toy:a",
                "step_index": step,
                "sigma": 1 - step * 0.5,
                "timestep": 1 - step * 0.5,
                "state_kind": "final" if step == 2 else "forward_input",
                "prompt": "edit",
                "task": "local_edit",
                "seed": 0,
                "height": 32,
                "width": 32,
                "state_representation": "toy",
            },
        )
        for step in range(3)
    ]
    TrajectoryCacheWriter(tmp_path).write(examples)
    manifest = {
        "schema_version": 2,
        "storage_layout": TRAJECTORY_GROUPED_LAYOUT,
        "model": "toy",
        "representation": "toy",
        "state_points": [0, 1, 2],
        "sources": 1,
        "records": 3,
        "creation_config": {
            "cache": {"num_steps": 2, "mask_variants": ["fine", "dilate5"]}
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETE").write_text("complete\n")
    return tmp_path, examples


@pytest.mark.parametrize("variant", ["default", "fine", "dilate5"])
def test_roundtrip_selects_mask_without_repeating_it_per_step(cache, variant):
    path, examples = cache
    assert validate_cache(path)["records"] == 3
    expected = (
        examples[0].keep_mask
        if variant == "default"
        else examples[0].mask_variants[variant]
    )
    with safe_open(next(path.glob("*.safetensors")), framework="pt") as handle:
        assert handle.get_slice("keep_mask_fine").get_shape() == [1, 128, 2, 2]
        assert handle.get_slice("keep_mask_dilate5").get_shape() == [1, 1, 2, 2]
    dataset = TrajectorySourceDataset(path, supervision_step=1, mask_variant=variant)
    batch = materialize_trajectory_state(
        trajectory_source_collate([dataset[0]]), TrajectoryPoint(0.5, 0.75, 0.75)
    )
    torch.testing.assert_close(batch["keep_mask"], expected.unsqueeze(0))
    segment = TrajectorySegmentDataset(path, supervision_step=1, mask_variant=variant)[
        0
    ]
    torch.testing.assert_close(segment.keep_mask, expected)
    # Existing readers continue using the original mask.
    torch.testing.assert_close(
        CacheDataset(path, steps=[1])[0].keep_mask, examples[0].keep_mask
    )


def test_missing_or_invalid_variant_is_never_silently_replaced(cache):
    path, _ = cache
    with pytest.raises(ValueError, match="unknown cache mask_variant"):
        TrajectorySourceDataset(path, supervision_step=1, mask_variant="typo")
    # Endpoint A now uses the same explicit variant as trajectory B/C/D.
    endpoint = compose_cache_datasets(
        [{"path": str(path), "mask_variant": "fine"}], steps=[1]
    )
    assert endpoint[0].metadata["mask_variant"] == "fine"
    assert endpoint[0].keep_mask.shape == endpoint[0].latent.shape
    shard = next(path.glob("*.safetensors"))
    tensors = load_file(shard)
    del tensors["keep_mask_fine"]
    save_file(tensors, shard)
    with pytest.raises(ValueError, match="lacks declared mask variant"):
        validate_cache(path)


def test_writer_rejects_wrong_shape_and_inconsistent_variants(cache, tmp_path):
    _, examples = cache
    bad = replace(
        examples[0], mask_variants={"fine": torch.ones(1, 2, 2, dtype=torch.uint8)}
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        TrajectoryCacheWriter(tmp_path / "bad-shape").write([bad])
    with pytest.raises(ValueError, match="mixes mask variants"):
        TrajectoryCacheWriter(tmp_path / "mixed").write(
            [examples[0], replace(examples[1], mask_variants=None)]
        )
