"""SenseNova trajectory native masks, on CPU and without model weights."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tailsteer.cache import trajectory
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
from tailsteer.models.masks import dilated_keep_mask
from tailsteer.models.protocols import TrajectoryPoint
from tailsteer.models.sensenova.cache import sensenova_keep_mask_variants
from tailsteer.models.sensenova.model import SenseNovaModel
from tailsteer.training.target_geometry import (
    InstantVelocityGeometry,
    TargetGeometryContext,
    TerminalBridgeGeometry,
)
from tailsteer.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def _masks(change):
    grid = (change.shape[0] // 32, change.shape[1] // 32)
    return sensenova_keep_mask_variants(
        change,
        state_shape=(1, grid[0] * grid[1], 3072),
        grid=grid,
        variants=["fine", "dilate5"],
    )


def test_fine_preserves_every_pixel_and_rgb_order():
    change = np.zeros((64, 96), dtype=bool)
    change[37, 73] = True  # Token (1,2), local pixel (5,9).
    masks = _masks(change)
    fine = masks["fine"]
    assert fine.shape == (1, 6, 3072) and fine.dtype == np.uint8
    assert (fine == 0).sum() == 3
    assert not fine[0, 5, (5 * 32 + 9) * 3 : (5 * 32 + 10) * 3].any()
    np.testing.assert_array_equal(
        masks["dilate5"], dilated_keep_mask(change, 2, 3, radius=5)
    )
    assert masks["dilate5"].shape == (1, 2, 3)
    assert not masks["dilate5"][0, 1, 2]


@pytest.mark.parametrize("task", ["local_edit", "noop"])
def test_builder_dispatches_native_masks_at_real_resolution(
    tmp_path, monkeypatch, task
):
    source = Image.new("RGB", (1024, 1024), "white")
    generated = source.copy()
    ImageDraw.Draw(generated).rectangle((250, 320, 393, 447), fill="black")
    source.save(tmp_path / "source.png")
    state = torch.zeros(1, 1024, 3072, dtype=torch.bfloat16)
    trace = trajectory.TrajectoryResult(
        image=generated,
        source_state=state,
        forward_states={1: state, 50: state},
        final_state=state,
        schedule_points={1: (1.0, 0.0), 50: (0.02, 0.98)},
        final_schedule_point=(0.0, 1.0),
        grid=(32, 32),
        extra={},
    )
    monkeypatch.setattr(trajectory, "_run_trajectory", lambda *_, **__: trace)
    config = load_config(ROOT / "configs/cache/sensenova-u1.5-8b-mot.yaml")
    config["cache"]["forward_points"] = [1, 50]
    _, examples, _ = trajectory.prepare_trajectory_examples(
        runtime=None,
        record={"id": "a", "source": "source.png", "prompt": "edit", "task": task},
        config=config,
        image_root=tmp_path,
    )
    assert len(examples) == 3
    first = examples[0]
    assert first.keep_mask.shape == (1, 32, 32)
    assert first.mask_variants["fine"].shape == state.shape
    assert first.mask_variants["dilate5"].shape == first.keep_mask.shape
    assert all(example.mask_variants is first.mask_variants for example in examples)
    if task == "noop":
        assert first.keep_mask.all()
        assert all(mask.all() for mask in first.mask_variants.values())
    else:
        assert (
            first.mask_variants["fine"].float().mean()
            > first.mask_variants["dilate5"].float().mean()
            > first.keep_mask.float().mean()
        )


@pytest.fixture
def cache(tmp_path):
    change = np.zeros((64, 64), dtype=bool)
    change[2, 3] = True
    masks = {name: torch.from_numpy(value) for name, value in _masks(change).items()}
    examples = [
        CacheExample(
            anchor_latent=torch.full((1, 4, 3072), float(step)),
            source_latent=torch.zeros(1, 4, 3072),
            keep_mask=torch.zeros(1, 2, 2, dtype=torch.uint8),
            mask_variants=masks,
            metadata={
                "sample_id": f"a:{step}",
                "source_id": "a",
                "source_identity": "test:a",
                "prompt": "edit",
                "task": "local_edit",
                "seed": 0,
                "height": 64,
                "width": 64,
                "step_index": step,
                "sigma": (3 - step) * 0.5,
                "timestep": (step - 1) * 0.5,
                "state_kind": "final" if step == 3 else "forward_input",
                "state_representation": "packed_rgb_pixel_tokens",
            },
        )
        for step in [1, 2, 3]
    ]
    TrajectoryCacheWriter(tmp_path).write(examples)
    manifest = {
        "schema_version": 2,
        "storage_layout": TRAJECTORY_GROUPED_LAYOUT,
        "model": "sensenova_u1_5_8b_mot",
        "representation": "packed_rgb_pixel_tokens",
        "state_points": [1, 2, 3],
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
def test_cache_roundtrip_and_training_shape(cache, variant):
    path, examples = cache
    validate_cache(path)
    with safe_open(next(path.glob("*.safetensors")), framework="pt") as handle:
        assert handle.get_slice("keep_mask_fine").get_shape() == [1, 1, 4, 3072]
        assert handle.get_slice("keep_mask_dilate5").get_shape() == [1, 1, 2, 2]
    expected = (
        examples[0].keep_mask
        if variant == "default"
        else examples[0].mask_variants[variant]
    )
    source = TrajectorySourceDataset(path, supervision_step=2, mask_variant=variant)[0]
    batch = materialize_trajectory_state(
        trajectory_source_collate([source]), TrajectoryPoint(1, 1, 0)
    )
    torch.testing.assert_close(batch["keep_mask"], expected.unsqueeze(0))
    segment = TrajectorySegmentDataset(path, supervision_step=2, mask_variant=variant)[
        0
    ]
    torch.testing.assert_close(segment.keep_mask, expected)
    mask = SenseNovaModel.tail_keep_mask(batch["keep_mask"], batch["latent"][:, 0])
    if variant == "fine":
        torch.testing.assert_close(mask, expected.to(torch.float32))
    else:
        torch.testing.assert_close(mask, expected.reshape(1, 4, 1).float())


@pytest.mark.parametrize(
    "geometry", [TerminalBridgeGeometry(), InstantVelocityGeometry()]
)
def test_fine_target_and_gradient_do_not_average_token_channels(geometry):
    change = np.zeros((32, 32), dtype=bool)
    change[0, 0] = True
    state = torch.zeros(1, 1, 3072)
    mask = SenseNovaModel.tail_keep_mask(
        torch.from_numpy(_masks(change)["fine"]).unsqueeze(0), state
    )
    context = TargetGeometryContext(
        current=state,
        source=torch.ones_like(state),
        keep=mask,
        point=TrajectoryPoint(50, 0.1, 0.9),
        following_point=None,
        reference=None,
        alpha=1.0,
        alpha_fn=lambda _: 1.0,
        preservation_scale=1.0,
        model=SenseNovaModel,
        terminal=state,
    )
    target = geometry.build(torch.zeros_like(state), context).velocity
    assert target[0, 0, :3].eq(0).all()
    assert target[0, 0, 3:].eq(10).all()
    student = torch.zeros_like(state, requires_grad=True)
    (student - target).square().mean().backward()
    assert student.grad[0, 0, :3].eq(0).all()
    assert student.grad[0, 0, 3:].lt(0).all()


def test_shape_errors_fail_closed(cache):
    path, _ = cache
    with pytest.raises(ValueError, match="grid differs"):
        SenseNovaModel.tail_keep_mask(torch.ones(1, 2, 3), torch.zeros(1, 4, 3072))
    with pytest.raises(ValueError, match="batch shapes"):
        SenseNovaModel.tail_keep_mask(torch.ones(2, 4, 3072), torch.zeros(1, 4, 3072))
    shard = next(path.glob("*.safetensors"))
    tensors = load_file(shard)
    tensors["keep_mask_dilate5"] = torch.ones_like(tensors["keep_mask_fine"])
    save_file(tensors, shard)
    with pytest.raises(ValueError, match="invalid dilate5 mask shape"):
        validate_cache(path)


def test_config_pins_same_candidates_native_steps_and_pixel_scale(tmp_path):
    config = load_config(ROOT / "configs/cache/sensenova-u1.5-8b-mot.yaml")
    flux = load_config(ROOT / "configs/cache/flux2-klein-base-9b.yaml")
    assert config["data"]["manifest"] == flux["data"]["manifest"]
    assert "manifest_sha256" not in config["data"]  # User-provided source manifest.
    assert config["cache"]["forward_points"] == [
        p + 1 for p in flux["cache"]["forward_points"]
    ]
    assert config["cache"]["final_point"] == 51
    assert config["model"]["variant"] == "sensenova-u1.5-8b-mot"
    for field in ("min_patch_side", "patch_multiple"):
        assert (
            config["cache"]["filter"][field] * 32 == flux["cache"]["filter"][field] * 16
        )
    config["model"].update(path=str(tmp_path), code_path=str(tmp_path))
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text("{}\n")
    config["data"].update(
        manifest=str(manifest),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    trajectory.validate_trajectory_config(config)
    assert config["cache"]["candidate_end"] == 1
    manifest.write_text("{}\n{}\n")
    with pytest.raises(ValueError, match="checksum changed"):
        trajectory.validate_trajectory_config(config)
