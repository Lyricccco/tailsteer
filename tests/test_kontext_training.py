"""Kontext contracts tested with the real BFL blocks at a small CPU width."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tailsteer.models.flux1_kontext.model import KontextModel
from tailsteer.models.flux1_kontext.official import import_source
from tailsteer.models.protocols import TailVelocityBranches

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    import os

    code = Path(os.environ.get("TAILSTEER_FLUX_CODE", REPOSITORY / "upstream/flux"))
    if not code.is_dir():
        pytest.skip("Install the pinned FLUX source with scripts/setup.py")
    import_source(code)
    from flux.model import Flux, FluxParams

    torch.manual_seed(1)
    transformer = Flux(
        FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=8,
            context_in_dim=12,
            hidden_size=24,
            mlp_ratio=2,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=[4, 4, 4],
            theta=10000,
            qkv_bias=True,
            guidance_embed=True,
        )
    )
    model = KontextModel(
        refined_transformer=transformer,
        pretrained_transformer=deepcopy(transformer),
        t5=lambda prompts: torch.ones(len(prompts), 3, 12),
        clip=lambda prompts: torch.ones(len(prompts), 8),
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_name="flux1_kontext_dev",
        model_path="/unused",
        layer_checkpointing=True,
    )
    model.configure_trainable({"policy": "denoiser_full", "master_dtype": "model"})
    return model


def test_native_velocity_euler_and_gradient_parity():
    model = tiny_model()
    from flux.sampling import denoise, prepare, unpack

    state, source = torch.randn(1, 16, 4, 4), torch.randn(1, 16, 4, 4)
    condition = model.prepare_conditioning_from_metadata(
        source_state=source,
        metadata=[{}],
        prompts=["blue"],
        height=32,
        width=32,
    )
    assert condition.payload["img_ids"].dtype == torch.float32
    assert torch.all(condition.payload["img_cond_seq_ids"][..., 0] == 1)
    schedule = model.tail_schedule(num_steps=30, height=32, width=32)
    point, following = schedule[27], schedule[28]
    velocity = model.predict_tail_velocity(state, point, condition, actor="refined")
    native = prepare(model.t5, model.clip, state, ["blue"])
    native.update(
        {k: condition.payload[k] for k in ("img_cond_seq", "img_cond_seq_ids")}
    )
    with torch.no_grad():
        expected = denoise(
            model.refined_transformer,
            **native,
            timesteps=[point.sigma, following.sigma],
            guidance=2.5,
        )
    actual = model.tail_advance(state, velocity, current=point, following=following)
    assert torch.equal(actual, unpack(expected, 32, 32))
    velocity.square().mean().backward()
    for name, parameter in model.refined_transformer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.pretrained_transformer.parameters()
    )


def test_native_bf16_schedule_and_update_are_not_fp32_euler():
    model = tiny_model()
    model.dtype = torch.bfloat16
    schedule = model.tail_schedule(num_steps=30, height=1024, width=1024)
    assert list(schedule) == list(range(31))
    assert schedule[29].sigma == pytest.approx(0.09820804)
    assert schedule[29].timestep == 0.09814453125
    assert schedule[30].sigma == 0
    current, following = schedule[27], schedule[28]
    state, velocity = torch.randn(1000).bfloat16(), torch.randn(1000).bfloat16()
    expected = state + (following.sigma - current.sigma) * velocity
    assert torch.equal(
        model.tail_advance(state, velocity, current=current, following=following),
        expected,
    )
    assert not torch.equal(
        expected,
        (
            state.float() + (following.sigma - current.sigma) * velocity.float()
        ).bfloat16(),
    )
    with pytest.raises(ValueError, match="one guidance-distilled"):
        model.compose_tail_velocity_branches(TailVelocityBranches({"base": velocity}))


def test_checkpointed_native_blocks_keep_gradients_and_export(tmp_path):
    model = tiny_model()
    model.prepare_activation_checkpointing()
    state = torch.randn(1, 16, 4, 4)
    condition = model.prepare_conditioning_from_metadata(
        source_state=state,
        metadata=[{}],
        prompts=["blue"],
        height=32,
        width=32,
    )
    point = model.tail_schedule(num_steps=30, height=32, width=32)[29]
    model.predict_tail_velocity(
        state, point, condition, actor="refined"
    ).sum().backward()
    assert all(p.grad is not None for p in model.trainable_parameters())
    weights, filename, manifest = model.prepare_inference_export(
        model.refined_transformer.state_dict(),
        step=1,
    )
    assert set(weights) == set(model._export_names)
    assert all(v.dtype == torch.bfloat16 for v in weights.values())
    assert filename == "transformer.safetensors"
    assert manifest["source_revision"] == model.source_revision
    assert len(model.auxiliary_sharding_groups()[0][2]) == 2


def test_kontext_t2i_matches_native_denoise_without_reference_tokens(monkeypatch):
    from contextlib import nullcontext
    from threading import RLock
    from types import SimpleNamespace

    from PIL import Image

    from tailsteer.models.flux1_kontext.inference import KontextInferenceRuntime

    model = tiny_model()
    model.refined_transformer.to(torch.bfloat16).eval()
    from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack

    runtime = SimpleNamespace(
        model=model.refined_transformer,
        device="cpu",
        _lock=RLock(),
        ae=None,
        t5=lambda prompts: model.t5(prompts).bfloat16(),
        clip=lambda prompts: model.clip(prompts).bfloat16(),
    )
    steps, decoded, token_counts = [], [], []
    actors = SimpleNamespace(
        use_policy=lambda _: nullcontext(),
        complete_step=lambda: steps.append(1),
        trace=("base",) * 4,
    )
    engine = KontextInferenceRuntime(runtime, actors, {"guidance_scale": 2.5})
    monkeypatch.setattr(
        engine,
        "materialize",
        lambda state: decoded.append(state) or Image.new("RGB", (64, 32)),
    )
    hook = runtime.model.register_forward_pre_hook(
        lambda module, args, kw: token_counts.append(kw["img"].shape[1]),
        with_kwargs=True,
    )
    image, trace = engine.generate_text(
        prompt="A bird",
        seed=42,
        policy=None,
        config={"height": 32, "width": 64, "num_steps": 4},
    )
    hook.remove()
    assert image.size == (64, 32) and len(trace) == len(steps) == 4
    assert token_counts == [8] * 4  # Target only; no dummy/reference image tokens.
    with torch.inference_mode():
        noise = get_noise(1, 32, 64, "cpu", torch.bfloat16, 42)
        inputs = prepare(runtime.t5, runtime.clip, noise, "A bird")
        expected = denoise(
            runtime.model, **inputs, timesteps=get_schedule(4, 8), guidance=2.5
        )
    assert torch.equal(decoded[0], unpack(expected, 32, 64)[0])
    assert not runtime.model._forward_hooks


@pytest.mark.parametrize("endpoint", [False, True])
def test_inference_native_loop_actor_switching_and_endpoint_skip(
    tmp_path, monkeypatch, endpoint
):
    import json
    import threading
    from types import SimpleNamespace

    from PIL import Image
    from safetensors.torch import save_file

    from tailsteer.inference.endpoint import build_endpoint_plans
    from tailsteer.inference.policies import InferencePolicy, PolicySegment
    from tailsteer.models.flux1_kontext.inference import KontextInferenceRuntime
    from tailsteer.models.native_transformer import TransformerActors

    model = tiny_model()
    from flux.sampling import prepare

    state = torch.randn(1, 16, 4, 4)

    def prepare_inputs(runtime, source, *, prompt, seed):
        value = prepare(runtime.t5, runtime.clip, state, prompt)
        ids = value["img_ids"].clone()
        ids[..., 0] = 1
        value.update(img_cond_seq=value["img"], img_cond_seq_ids=ids)
        return value, 32, 32

    monkeypatch.setattr(
        "tailsteer.models.flux1_kontext.inference.prepare_shared_inputs", prepare_inputs
    )
    monkeypatch.setattr(
        "tailsteer.models.flux1_kontext.inference.decode_image",
        lambda *args, **kwargs: Image.new("RGB", (32, 32)),
    )
    weights, filename, manifest = model.prepare_inference_export(
        model.refined_transformer.state_dict(), step=1
    )
    save_file(weights, tmp_path / filename)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETE").touch()
    actors = TransformerActors(
        model.refined_transformer,
        tmp_path,
        model_name=model.model_name,
        export_format=model.export_format,
    )
    native = SimpleNamespace(
        model=model.refined_transformer,
        t5=model.t5,
        clip=model.clip,
        ae=None,
        device=model.device,
        _lock=threading.RLock(),
    )
    runtime = KontextInferenceRuntime(native, actors, {"guidance_scale": 2.5})
    inference = {"num_steps": 30, "feedback_mode": "image", "input_preprocessed": True}
    if endpoint:
        plan = build_endpoint_plans(
            num_steps=30, target_step=27, target_actor_index=27
        )["transition"]
        policy = plan.actor_policy()
        inference["endpoint_plan"] = plan.to_dict()
        expected = ("base",) * 27 + ("refined",)
    else:
        policy = InferencePolicy(
            name="transition_refined",
            num_steps=30,
            segments=(
                PolicySegment(0, 27, "base"),
                PolicySegment(27, 30, "refined"),
            ),
        )
        expected = ("base",) * 27 + ("refined",) * 3
    result = runtime.generate(
        Image.new("RGB", (32, 32)),
        prompt="blue",
        seed=0,
        policy=policy,
        inference=inference,
    )
    assert result.trace == expected
    assert result.state.shape == (16, 4, 4)
    assert actors.state.actor == "base"
    assert not native.model._forward_hooks
