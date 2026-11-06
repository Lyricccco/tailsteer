"""Kontext trajectory uses spatial VAE masks, never FLUX.2 packed-channel masks."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from tailsteer.cache import trajectory
from tailsteer.cache.dataset import CacheDataset
from tailsteer.cache.format import TrajectoryCacheWriter, validate_cache
from tailsteer.cache.trajectory_dataset import (
    TrajectorySegmentDataset,
    TrajectorySourceDataset,
    materialize_trajectory_state,
    trajectory_source_collate,
)
from tailsteer.models.flux1_kontext.cache import keep_mask_variants
from tailsteer.models.flux1_kontext.model import KontextModel
from tailsteer.models.masks import project_keep_mask
from tailsteer.models.protocols import TrajectoryPoint
from tailsteer.utils import load_config


def test_fine_keeps_vae_subpositions_spatial_and_dilate5_uses_image_pixels():
    change = np.zeros((64, 64), dtype=bool)
    change[28, 28] = True
    masks = keep_mask_variants(
        change, state_shape=(16, 8, 8), variants=["fine", "dilate5"]
    )
    assert all(
        mask.shape == (1, 8, 8) and mask.dtype == np.uint8 for mask in masks.values()
    )
    np.testing.assert_array_equal(masks["fine"][0], project_keep_mask(change, 8, 8))
    assert (masks["fine"] == 0).sum() == 1
    assert not masks["fine"][0, 3, 3]
    assert masks["fine"][0, 2, 3] and masks["fine"][0, 3, 4]
    assert not masks["dilate5"][0, 2, 3]  # 28-5=23 reaches the preceding VAE cell.
    assert not masks["dilate5"][0, 3, 4]  # 28+5=33 reaches the following VAE cell.
    assert np.all(masks["dilate5"] <= masks["fine"])


@pytest.mark.parametrize(
    "shape,variants",
    [((64, 4, 4), ["fine"]), ((16, 4, 4), ["fine"]), ((16, 8, 8), ["typo"])],
)
def test_mask_rejects_packed_or_wrong_resolution(shape, variants):
    with pytest.raises(ValueError):
        keep_mask_variants(
            np.zeros((64, 64), dtype=bool), state_shape=shape, variants=variants
        )


@pytest.fixture(params=["local_edit", "noop"])
def cache(tmp_path, monkeypatch, request):
    task = request.param
    source = Image.new("RGB", (1024, 1024), "white")
    source.save(tmp_path / "source.png")
    generated = source.copy()
    if task == "local_edit":
        ImageDraw.Draw(generated).rectangle((250, 320, 393, 447), fill="black")
    state = torch.zeros(16, 128, 128, dtype=torch.bfloat16)
    trace = trajectory.TrajectoryResult(
        image=generated,
        source_state=state,
        final_state=state,
        forward_states={0: state, 29: state},
        schedule_points={0: (1.0, 1.0), 29: (0.1, 0.10009765625)},
        final_schedule_point=(0.0, 0.0),
        grid=(128, 128),
        extra={},
    )
    monkeypatch.setattr(trajectory, "_run_trajectory", lambda *_, **__: trace)
    config = load_config("configs/cache/flux1-kontext-dev.yaml")
    config["cache"]["forward_points"] = [0, 29]
    kwargs = dict(
        runtime=None,
        config=config,
        image_root=tmp_path,
        record={"id": "a", "source": "source.png", "prompt": "edit", "task": task},
    )
    _, examples, _ = trajectory.prepare_trajectory_examples(**kwargs)
    assert len(examples) == 3  # Real numeric filter is enabled.
    first = examples[0]
    assert first.keep_mask.shape == (1, 128, 128)
    assert all(
        mask.shape == first.keep_mask.shape for mask in first.mask_variants.values()
    )
    assert all(item.mask_variants is first.mask_variants for item in examples)
    config["cache"]["mask_variants"] = []
    _, legacy, _ = trajectory.prepare_trajectory_examples(**kwargs)
    torch.testing.assert_close(first.keep_mask, legacy[0].keep_mask)
    if task == "noop":
        assert first.keep_mask.all() and all(
            mask.all() for mask in first.mask_variants.values()
        )
    else:
        assert (
            first.mask_variants["fine"].float().mean()
            > first.mask_variants["dilate5"].float().mean()
            > first.keep_mask.float().mean()
        )
    TrajectoryCacheWriter(tmp_path).write(examples)
    manifest = {
        "schema_version": 2,
        "storage_layout": "trajectory_grouped_v2",
        "model": "flux1_kontext_dev",
        "representation": "flux1_unpacked_vae_latent",
        "state_points": [0, 29, 30],
        "sources": 1,
        "records": 3,
        "creation_config": {
            "cache": {"num_steps": 30, "mask_variants": ["fine", "dilate5"]}
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETE").write_text("complete\n")
    return tmp_path, examples


@pytest.mark.parametrize("variant", ["default", "fine", "dilate5"])
def test_grouped_roundtrip_and_training_readers(cache, variant):
    path, examples = cache
    assert validate_cache(path)["records"] == 3
    expected = (
        examples[0].keep_mask
        if variant == "default"
        else examples[0].mask_variants[variant]
    )
    endpoint = CacheDataset(
        path, steps=[29], mask_variant=variant, include_terminal_state=True
    )[0]
    torch.testing.assert_close(endpoint.keep_mask, expected)
    dataset = TrajectorySourceDataset(path, supervision_step=29, mask_variant=variant)
    batch = materialize_trajectory_state(
        trajectory_source_collate([dataset[0]]), TrajectoryPoint(0.0, 1.0, 1.0)
    )
    torch.testing.assert_close(batch["keep_mask"], expected[None])
    torch.testing.assert_close(
        TrajectorySegmentDataset(path, supervision_step=29, mask_variant=variant)[
            0
        ].keep_mask,
        expected,
    )
    state = torch.ones(1, 16, 128, 128, requires_grad=True)
    mask = KontextModel.tail_keep_mask(batch["keep_mask"], state)
    (state * mask).sum().backward()
    torch.testing.assert_close(state.grad, mask.expand_as(state))


def test_only_kontext_allows_compact_fine_mask(cache, tmp_path):
    path, examples = cache
    wrong = replace(
        examples[0],
        metadata={
            **examples[0].metadata,
            "state_representation": "flux2_unpacked_vae_latent",
        },
    )
    with pytest.raises(ValueError, match="invalid fine mask shape"):
        TrajectoryCacheWriter(tmp_path / "wrong").write([wrong])
    shard = next(path.glob("*.safetensors"))
    tensors = load_file(shard)
    tensors["keep_mask_fine"] = (
        tensors["keep_mask_fine"].expand(1, 16, 128, 128).contiguous()
    )
    save_file(tensors, shard)
    with pytest.raises(ValueError, match="invalid fine mask shape"):
        validate_cache(path)
