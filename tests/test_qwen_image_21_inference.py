"""Native-size Qwen inference; scheduler tests use the pinned qwen21 environment."""

from contextlib import contextmanager
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tailsteer.inference.policies import InferencePolicy, PolicySegment
from tailsteer.models.qwen_image_21.inference import (
    QwenImage21InferenceRuntime,
    schedule_context,
)
from tailsteer.models.qwen_image_21.preprocessing import (
    prepare_source,
    reference_canvas_context,
    source_geometry,
)


@pytest.mark.parametrize(
    "size,expected",
    [
        ((1024, 683), (1024, 672)),
        ((685, 1024), (672, 1024)),
        ((2040, 1356), (2016, 1344)),
        ((1024, 1024), (1024, 1024)),
        ((1056, 672), (1056, 672)),
    ],
)
def test_native_crop_is_only_alignment_and_is_idempotent(size, expected):
    pixels = np.random.default_rng(0).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    geometry = source_geometry(size)
    output = prepare_source(source)
    assert output.size == expected
    assert source.crop(geometry["crop_box"]).tobytes() == output.tobytes()
    assert prepare_source(output).tobytes() == output.tobytes()
    assert geometry["generation_size"] == list(expected)
    assert geometry["reference_size"] == list(expected)
    assert size[0] - expected[0] < 32 and size[1] - expected[1] < 32
    assert prepare_source(output, input_preprocessed=True).tobytes() == output.tobytes()


@pytest.mark.parametrize("size", [(16, 1024), (1024, 16), (1024, 683)])
def test_prepared_canvas_cannot_be_recropped(size):
    with pytest.raises(ValueError):
        prepare_source(Image.new("RGB", size), input_preprocessed=True)


@pytest.fixture
def official():
    return pytest.importorskip("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")


def _scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
        shift_terminal=0.02,
    )


def _set_schedule(official, scheduler, width, height):
    cfg = scheduler.config
    mu = official.calculate_shift(
        width * height // 256,
        cfg.base_image_seq_len,
        cfg.max_image_seq_len,
        cfg.base_shift,
        cfg.max_shift,
    )
    official.retrieve_timesteps(
        scheduler, 40, "cpu", sigmas=np.linspace(1, 1 / 40, 40), mu=mu
    )


@pytest.mark.parametrize("policy", [None, "native", "fixed_reference_shape"])
def test_schedule_uses_reference_only_when_requested(official, policy):
    scheduler, expected = _scheduler(), _scheduler()
    schedule = {} if policy is None else {"shift_policy": policy}
    with schedule_context(scheduler, schedule):
        _set_schedule(official, scheduler, 1024, 672)
    _set_schedule(
        official, expected, 1024, 1024 if policy == "fixed_reference_shape" else 672
    )
    torch.testing.assert_close(scheduler.sigmas, expected.sigmas, rtol=0, atol=0)
    assert "set_timesteps" not in vars(scheduler)
    assert len(scheduler.timesteps) == 40 and len(scheduler.sigmas) == 41


def test_schedule_restored_on_exception(official):
    scheduler = _scheduler()
    schedule = {"shift_policy": "fixed_reference_shape"}
    with (
        pytest.raises(RuntimeError, match="probe"),
        schedule_context(scheduler, schedule),
    ):
        raise RuntimeError("probe")
    assert "set_timesteps" not in vars(scheduler)
    override = scheduler.set_timesteps
    scheduler.set_timesteps = override
    with schedule_context(scheduler, schedule):
        _set_schedule(official, scheduler, 1024, 672)
    assert scheduler.set_timesteps is override


def test_default_schedule_depends_on_actual_canvas(official):
    schedules = []
    for height in (672, 1024):
        scheduler = _scheduler()
        with schedule_context(scheduler, {}):
            assert "set_timesteps" not in vars(scheduler)
            _set_schedule(official, scheduler, 1024, height)
        schedules.append(scheduler.sigmas)
    assert not torch.equal(*schedules)


@pytest.mark.parametrize("size", [(1024, 683), (685, 1024), (1056, 672), None])
@pytest.mark.parametrize("reference_count", [0, 2])
def test_generate_preserves_reference_canvas_and_unpacks_actual_shape(
    official, size, reference_count
):
    references = tuple(
        Image.new("RGB", (128, 96), (i + 1,) * 3) for i in range(reference_count)
    )

    class Actors:
        trace = ()

        @contextmanager
        def use_policy(self, policy):
            self.policy, self.trace = policy, ()
            yield

        def complete_step(self):
            self.trace += (self.policy.actor_at(len(self.trace)),)

    class Pipeline:
        transformer = torch.nn.Linear(1, 1)
        scheduler = _scheduler()
        processor = SimpleNamespace(image_processor=SimpleNamespace(do_resize=True))

        def __call__(self, **kwargs):
            source = kwargs["image"]
            if source is not None and reference_count:
                assert isinstance(source, list)
                assert source[1:] == list(references)
                source = source[0]
            width, height = kwargs["width"], kwargs["height"]
            if source is not None:
                assert self.processor.image_processor.do_resize is False
                ref = official.calculate_dimensions(
                    kwargs["output_resolution"] ** 2, source.width / source.height
                )[:2]
                assert ref == source.size == (width, height)
            else:
                assert (width, height) == (1024, 672)
            assert kwargs["num_inference_steps"] == 40
            assert kwargs["true_cfg_scale"] == 1.0
            _set_schedule(official, self.scheduler, width, height)
            state = torch.zeros(1, width * height // 256, 64, dtype=torch.bfloat16)
            for step in range(40):
                kwargs["callback_on_step_end"](self, step, 0, {"latents": state})
            return SimpleNamespace(images=[Image.new("RGBA", (width, height), "red")])

    runtime = QwenImage21InferenceRuntime()
    runtime.runtime = SimpleNamespace(pipeline=Pipeline(), device="cpu", _lock=RLock())
    runtime.model, runtime.actors = {}, Actors()
    policy = InferencePolicy(
        "transition",
        [PolicySegment(0, 30, "base"), PolicySegment(30, 40, "refined")],
        40,
    )
    if size is None:
        image, trace = runtime.generate_text(
            prompt="generate",
            seed=42,
            policy=policy,
            config={"num_steps": 40, "width": 1024, "height": 672},
        )
        assert image.mode == "RGB" and image.size == (1024, 672)
        assert trace == tuple(["base"] * 30 + ["refined"] * 10)
        assert runtime.conditioning_actor_for(policy, num_steps=40) == "base"
        return
    source = Image.new("RGB", size, "red")
    for _ in range(2):
        result = runtime.generate(
            source,
            prompt="edit",
            seed=0,
            policy=policy,
            inference={"num_steps": 40},
            condition_images=references,
        )
        if reference_count:
            assert result.metadata["input_image_count"] == reference_count + 1
        width, height = result.image.size
        assert result.state.shape == (64, height // 16, width // 16)
        assert (width, height) == (size[0] // 32 * 32, size[1] // 32 * 32)
        assert result.trace == tuple(["base"] * 30 + ["refined"] * 10)
        expected = _scheduler()
        _set_schedule(official, expected, width, height)
        torch.testing.assert_close(
            runtime.runtime.pipeline.scheduler.sigmas, expected.sigmas, rtol=0, atol=0
        )
        source = result.image
    assert "set_timesteps" not in vars(runtime.runtime.pipeline.scheduler)
    assert runtime.runtime.pipeline.processor.image_processor.do_resize is True


def test_reference_resize_restored_after_failure():
    pipe = SimpleNamespace(
        processor=SimpleNamespace(image_processor=SimpleNamespace(do_resize=True))
    )
    with pytest.raises(RuntimeError), reference_canvas_context(pipe):
        assert pipe.processor.image_processor.do_resize is False
        raise RuntimeError("encoding failed")
    assert pipe.processor.image_processor.do_resize is True
