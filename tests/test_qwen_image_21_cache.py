"""CPU contract checks; the opt-in GPU probe checks the actual official forward."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open

from tailsteer.cache import trajectory
from tailsteer.cache.format import (
    TRAJECTORY_GROUPED_LAYOUT,
    CacheExample,
    TrajectoryCacheWriter,
    validate_cache,
)
from tailsteer.cache.trajectory_dataset import (
    TrajectorySourceDataset,
    materialize_trajectory_state,
    trajectory_source_collate,
)
from tailsteer.models.protocols import TrajectoryPoint
from tailsteer.models.qwen_image_21.official import (
    capture_states,
    composite_rgb,
    keep_mask_variants,
    unpack_state,
)
from tailsteer.utils import load_config

CONFIG = "configs/cache/qwen-image-2.1.yaml"


class Scheduler:
    def __init__(self):
        self.calls = 0

    def step(self, velocity, timestep, sample, return_dict=False):
        k = self.calls
        assert timestep == self.timesteps[k]
        self.calls += 1
        output = sample.float() + (self.sigmas[k + 1] - self.sigmas[k]) * velocity
        return (output,) if not return_dict else SimpleNamespace(prev_sample=output)


class Pipeline:
    def __init__(self):
        self.scheduler = Scheduler()

    def prepare_latents(self):
        return (
            torch.full((1, 4, 64), 2, dtype=torch.bfloat16),
            torch.full((1, 4, 64), 7, dtype=torch.bfloat16),
        )

    def run(self, *, return_dict=False):
        latents, _ = self.prepare_latents()
        # As in the real pipeline, scheduling happens AFTER latent preparation.
        self.scheduler.sigmas = torch.tensor([1.0, 0.5, 0.02, 0.0])
        self.scheduler.timesteps = self.scheduler.sigmas[:-1] * 1000
        for t in self.scheduler.timesteps:
            result = self.scheduler.step(
                torch.ones_like(latents), t, latents, return_dict=return_dict
            )
            latents = result.prev_sample if return_dict else result[0]
        return latents


@pytest.mark.parametrize("return_dict", [False, True])
def test_capture_does_not_change_native_sampler_or_precision(return_dict):
    expected = Pipeline().run(return_dict=return_dict)
    pipe = Pipeline()
    with capture_states(pipe, points=(0, 2), num_steps=3, width=32, height=32) as t:
        actual = pipe.run(return_dict=return_dict)
    assert torch.equal(actual, expected)
    assert t.source_state.shape == (64, 2, 2)
    assert t.source_state.eq(7).all()
    assert t.forward_states[0].dtype == torch.bfloat16
    assert t.forward_states[2].dtype == t.final_state.dtype == torch.float32
    torch.testing.assert_close(t.final_state, torch.ones(64, 2, 2))
    assert t.schedule_points[2] == (float(pipe.scheduler.sigmas[2]), 20.0)
    assert t.final_schedule_point == (0.0, 0.0)
    assert set(t.forward_velocities) == {0, 2}
    actual.zero_()
    assert t.final_state.eq(1).all()
    assert "prepare_latents" not in pipe.__dict__
    assert "step" not in pipe.scheduler.__dict__


@pytest.mark.parametrize("points", [(), (1, 1), (2, 1), (-1,), (3,)])
def test_invalid_capture_points_fail_before_forward(points):
    with (
        pytest.raises(ValueError),
        capture_states(Pipeline(), points=points, num_steps=3, width=32, height=32),
    ):
        pytest.fail("should not reach model")


def test_capture_restores_overrides_after_interruption():
    pipe = Pipeline()
    prepare, step = pipe.prepare_latents, pipe.scheduler.step
    pipe.prepare_latents, pipe.scheduler.step = prepare, step
    with (
        pytest.raises(RuntimeError, match="interrupt"),
        capture_states(pipe, points=(0,), num_steps=3, width=32, height=32),
    ):
        pipe.prepare_latents()
        raise RuntimeError("interrupt")
    assert pipe.prepare_latents is prepare and pipe.scheduler.step is step
    with (
        pytest.raises(RuntimeError, match="incomplete"),
        capture_states(pipe, points=(0,), num_steps=3, width=32, height=32),
    ):
        pipe.prepare_latents()


def test_unflatten_and_rgba_display_are_explicit():
    tokens = torch.arange(256).reshape(1, 4, 64)
    state = unpack_state(tokens, width=32, height=32)
    assert state[3, 1, 0] == tokens[0, 2, 3]
    with pytest.raises(ValueError):
        unpack_state(tokens.repeat(1, 2, 1), width=32, height=32)
    transparent = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    assert composite_rgb(transparent).getpixel((0, 0)) == (255, 255, 255)
    opaque = Image.new("RGBA", (1, 1), (10, 20, 30, 255))
    assert composite_rgb(opaque).getpixel((0, 0)) == (10, 20, 30)


def test_masks_use_vae_cells_not_flux_packed_subpositions():
    change = np.zeros((64, 64), dtype=bool)
    change[16, 16] = True
    masks = keep_mask_variants(
        change, state_shape=(64, 4, 4), variants=["fine", "dilate5"]
    )
    assert masks["fine"].shape == (64, 4, 4)
    assert masks["fine"].dtype == np.uint8
    assert (masks["fine"] == 0).sum() == 64
    assert not masks["fine"][:, 1, 1].any()
    assert masks["dilate5"].shape == (1, 4, 4)
    assert masks["dilate5"].mean() < masks["fine"].mean()
    assert all(np.array_equal(channel, masks["fine"][0]) for channel in masks["fine"])


@pytest.mark.parametrize("task", ["local_edit", "noop"])
def test_common_builder_dispatches_qwen_and_preserves_three_masks(
    tmp_path, monkeypatch, task
):
    source = Image.new("RGB", (1024, 1024), "white")
    source.save(tmp_path / "source.png")
    rgba = source.convert("RGBA")
    ImageDraw.Draw(rgba).rectangle((250, 320, 393, 447), fill=(0, 0, 0, 255))
    state = torch.zeros(64, 64, 64, dtype=torch.bfloat16)
    trace = SimpleNamespace(
        image=rgba.convert("RGB"),
        rgba_image=rgba,
        source_state=state,
        forward_states={0: state, 39: state},
        forward_velocities={0: state, 39: state},
        final_state=state,
        schedule_points={0: (1, 1000), 39: (0.02, 20)},
        final_schedule_point=(0, 0),
    )
    received = {}

    def edit(**kwargs):
        received.update(kwargs)
        return trace

    cfg = load_config(CONFIG)
    cfg["cache"]["forward_points"] = [0, 39]
    _, examples, _ = trajectory.prepare_trajectory_examples(
        runtime=SimpleNamespace(edit=edit),
        record={"id": "a", "source": "source.png", "prompt": "edit", "task": task},
        config=cfg,
        image_root=tmp_path,
    )
    assert received["true_cfg_scale"] == 1 and received["use_kv_cache"] is True
    assert received["num_steps"] == 40
    assert len(examples) == 3
    masks = examples[0].mask_variants
    assert masks["fine"].shape == (64, 64, 64)
    assert masks["dilate5"].shape == examples[0].keep_mask.shape == (1, 64, 64)
    assert examples[0].metadata["output_alpha_min"] == 255
    assert (
        examples[0].metadata["state_representation"] == "qwen21_normalized_vae_latent"
    )
    if task == "noop":
        assert all(mask.all() for mask in masks.values())
    else:
        assert masks["fine"].float().mean() > examples[0].keep_mask.float().mean()


@pytest.mark.parametrize("variant", ["default", "fine", "dilate5"])
def test_grouped_roundtrip_uses_existing_dataset_selector(tmp_path, variant):
    change = np.zeros((64, 64), dtype=bool)
    change[20:25, 25:30] = True
    masks = {
        k: torch.from_numpy(v)
        for k, v in keep_mask_variants(
            change, state_shape=(64, 4, 4), variants=["fine", "dilate5"]
        ).items()
    }
    coarse = torch.ones(1, 4, 4, dtype=torch.uint8)
    examples = [
        CacheExample(
            anchor_latent=torch.zeros(64, 4, 4),
            source_latent=torch.ones(64, 4, 4),
            keep_mask=coarse,
            mask_variants=masks,
            metadata={
                "sample_id": f"a:{k}",
                "source_id": "a",
                "source_identity": "test:a",
                "prompt": "edit",
                "seed": 0,
                "task": "local_edit",
                "height": 64,
                "width": 64,
                "step_index": k,
                "sigma": 1 - k / 2,
                "timestep": 1000 - k * 500,
                "state_kind": "final" if k == 2 else "forward_input",
            },
        )
        for k in (0, 1, 2)
    ]
    TrajectoryCacheWriter(tmp_path).write(examples)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "storage_layout": TRAJECTORY_GROUPED_LAYOUT,
                "model": "qwen_image_21",
                "state_points": [0, 1, 2],
                "sources": 1,
                "records": 3,
                "creation_config": {"cache": {"num_steps": 2}},
            }
        )
    )
    (tmp_path / "COMPLETE").write_text("complete\n")
    validate_cache(tmp_path)
    with safe_open(next(tmp_path.glob("*.safetensors")), framework="pt") as f:
        assert f.get_slice("keep_mask_fine").get_shape() == [1, 64, 4, 4]
    row = TrajectorySourceDataset(tmp_path, supervision_step=1, mask_variant=variant)[0]
    expected = coarse if variant == "default" else masks[variant]
    batch = materialize_trajectory_state(
        trajectory_source_collate([row]), TrajectoryPoint(0, 1, 1000)
    )
    torch.testing.assert_close(batch["keep_mask"], expected.unsqueeze(0))


def test_config_pins_native_schedule_and_same_trajectory_manifest(tmp_path):
    cfg = load_config(CONFIG)
    assert cfg["cache"]["forward_points"] == list(range(0, 30, 3)) + list(range(30, 40))
    assert cfg["cache"]["final_point"] == cfg["cache"]["num_steps"] == 40
    assert (
        cfg["data"]["manifest"]
        == load_config("configs/cache/flux2-klein-base-9b.yaml")["data"]["manifest"]
    )
    cfg["model"].update(path=str(tmp_path), code_path=str(tmp_path))
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text("{}\n")
    cfg["data"].update(
        manifest=str(manifest),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    trajectory.validate_trajectory_config(cfg)
    cfg["cache"]["num_steps"] = 50
    with pytest.raises(ValueError, match="requires num_steps=40"):
        trajectory.validate_trajectory_config(cfg)
