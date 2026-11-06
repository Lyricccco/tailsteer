from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from tailsteer.inference.multiround import balanced_task_indices, load_tasks
from tailsteer.inference.policies import InferencePolicy, PolicySegment
from tailsteer.inference.sensenova_runner import (
    _prepare_model_input,
    _run_segmented_model_native_task,
    _run_task,
    sensenova_native_resolution,
)
from tailsteer.models.sensenova.conditioning import (
    PrefixKVAdapterConfig,
    attach_prefix_kv_adapter,
)
from tailsteer.models.sensenova.inference import (
    SenseNovaGenerationActors,
    SenseNovaPolicyInference,
    SenseNovaReferenceState,
)
from tailsteer.models.sensenova.official import (
    SenseNovaRuntime,
    run_sensenova_edit,
)


class _ToyRawModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fm_modules = torch.nn.ModuleDict(
            {
                "vision_model_mot_gen": torch.nn.Linear(1, 1, bias=False),
            }
        )
        self.fm_modules["vision_model_mot_gen"].weight.data.fill_(1)

    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        del gen_model, grid_hw
        return self.fm_modules["vision_model_mot_gen"](pixel_values)


def _actor_export(tmp_path) -> None:
    save_file(
        {"model.fm_modules.vision_model_mot_gen.weight": torch.tensor([[3.0]])},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "tailsteer_sensenova_delta",
                "weights": "model.safetensors",
                "step": 2500,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "COMPLETE").write_text("complete\n", encoding="utf-8")


def test_sensenova_actor_switches_only_at_late_activation(tmp_path) -> None:
    _actor_export(tmp_path)
    raw = _ToyRawModel()
    actors = SenseNovaGenerationActors(raw, tmp_path)
    value = torch.ones((1, 1))

    actors.begin("late_step", num_steps=3, activation_step=2)
    outputs = [raw.extract_feature(value, gen_model=True).item() for _ in range(3)]
    trace = actors.finish()

    assert outputs == [1.0, 1.0, 3.0]
    assert trace == ("base", "base", "refined")


def test_sensenova_all_steps_uses_refined_actor_everywhere(tmp_path) -> None:
    _actor_export(tmp_path)
    raw = _ToyRawModel()
    actors = SenseNovaGenerationActors(raw, tmp_path)

    actors.begin("all_steps", num_steps=3)
    outputs = [
        raw.extract_feature(torch.ones((1, 1)), gen_model=True).item() for _ in range(3)
    ]

    assert actors.finish() == ("refined", "refined", "refined")
    assert outputs == [3.0, 3.0, 3.0]


def test_sensenova_accepts_arbitrary_segmented_actor_policy(tmp_path) -> None:
    _actor_export(tmp_path)
    raw = _ToyRawModel()
    actors = SenseNovaGenerationActors(raw, tmp_path)
    policy = InferencePolicy(
        name="transition_refined",
        segments=(
            PolicySegment(start=0, end=1, actor="base"),
            PolicySegment(start=1, end=3, actor="refined"),
        ),
        num_steps=3,
    )

    actors.begin(policy, num_steps=3)
    outputs = [
        raw.extract_feature(torch.ones((1, 1)), gen_model=True).item() for _ in range(3)
    ]

    assert actors.finish() == ("base", "refined", "refined")
    assert outputs == [1.0, 3.0, 3.0]


class _ToyConditionedRawModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        attention = torch.nn.Module()
        attention.head_dim = 1
        attention.config = SimpleNamespace(num_key_value_heads=1)
        attention.q_proj = torch.nn.Linear(1, 1, bias=False)
        layer = torch.nn.Module()
        layer.self_attn = attention
        backbone = torch.nn.Module()
        backbone.layers = torch.nn.ModuleList([layer])
        self.language_model = torch.nn.Module()
        self.language_model.model = backbone
        self.fm_modules = torch.nn.ModuleDict(
            {
                "vision_model_mot_gen": torch.nn.Linear(1, 1, bias=False),
            }
        )
        self.vision_model = torch.nn.Linear(1, 1, bias=False)
        self.fm_modules["vision_model_mot_gen"].weight.data.fill_(1)
        self.vision_model.weight.data.fill_(1)
        self.device = torch.device("cpu")
        self.img_context_token_id = 7

    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        del grid_hw
        module = (
            self.fm_modules["vision_model_mot_gen"] if gen_model else self.vision_model
        )
        return module(pixel_values)

    def _build_it2i_inputs(self, tokenizer, query, pixel_values=None, grid_hw=None):
        del grid_hw
        input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
        embeddings = torch.zeros((*input_ids.shape, 1))
        if pixel_values is not None:
            reference = self.extract_feature(pixel_values)
            mask = input_ids.reshape(-1).eq(self.img_context_token_id)
            embeddings.reshape(-1, 1)[mask] = reference.reshape(-1, 1)
        return embeddings, None, None

    @staticmethod
    def _it2i_prefix_forward(input_embeddings, *_args):
        layer = SimpleNamespace(
            keys=input_embeddings.unsqueeze(1).clone(),
            values=input_embeddings.unsqueeze(1).clone(),
        )
        return SimpleNamespace(layers=[layer]), input_embeddings


class _ToyTokenizer:
    @staticmethod
    def __call__(_query, return_tensors="pt"):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[7, 2]])}


def _conditioned_actor_export(tmp_path, raw) -> None:
    config = PrefixKVAdapterConfig(
        enabled=True,
        target="image_tokens",
        last_n_layers=1,
        rank=1,
        scale=1.0,
    )
    adapter = attach_prefix_kv_adapter(raw, config)
    assert adapter is not None
    state = {
        "model.fm_modules.vision_model_mot_gen.weight": torch.tensor([[3.0]]),
        "model.vision_model.weight": torch.tensor([[5.0]]),
    }
    for name, parameter in adapter.named_parameters():
        state[f"model.tailsteer_prefix_kv_adapter.{name}"] = torch.ones_like(parameter)
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": "tailsteer_sensenova_delta",
                "weights": "model.safetensors",
                "step": 1000,
                "trainable": {
                    "policy": "generation_denoiser",
                    "reference_vision_model": True,
                    "prefix_kv_adapter": config.to_dict(),
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "COMPLETE").write_text("complete\n", encoding="utf-8")


def test_sensenova_schema_v2_switches_reference_and_prefix_adapter(tmp_path) -> None:
    raw = _ToyConditionedRawModel()
    _conditioned_actor_export(tmp_path, raw)
    actors = SenseNovaGenerationActors(raw, tmp_path)
    tokenizer = _ToyTokenizer()
    pixels = torch.ones((1, 1))

    actors.begin("base", num_steps=1)
    inputs, _, _ = raw._build_it2i_inputs(tokenizer, "source", pixels)
    pretrained_cache, _ = raw._it2i_prefix_forward(inputs, None, None)
    pretrained_generation = raw.extract_feature(pixels, gen_model=True)
    assert actors.finish() == ("base",)

    actors.begin("all_steps", num_steps=1)
    inputs, _, _ = raw._build_it2i_inputs(tokenizer, "source", pixels)
    refined_cache, _ = raw._it2i_prefix_forward(inputs, None, None)
    refined_generation = raw.extract_feature(pixels, gen_model=True)
    assert actors.finish() == ("refined",)

    assert pretrained_generation.item() == 1.0
    assert refined_generation.item() == 3.0
    assert pretrained_cache.layers[0].keys[0, 0, 0, 0].item() == 1.0
    assert refined_cache.layers[0].keys[0, 0, 0, 0].item() == 15.0
    assert actors.generation_parameter_count == 1
    assert actors.conditioning_parameter_count == 7


def test_sensenova_schema_v3_switches_complete_conditioning_backbone(tmp_path) -> None:
    raw = _ToyConditionedRawModel()
    state = {
        f"model.{name}": torch.full_like(parameter, 3.0 + index)
        for index, (name, parameter) in enumerate(raw.named_parameters())
    }
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "format": "tailsteer_sensenova_delta",
                "weights": "model.safetensors",
                "step": 1000,
                "trainable": {
                    "policy": "full_model",
                    "reference_vision_model": False,
                    "prefix_kv_adapter": {"enabled": False},
                    "conditioning_gradients": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "COMPLETE").write_text("complete\n", encoding="utf-8")

    actors = SenseNovaGenerationActors(raw, tmp_path)
    ordinary = raw.language_model.model.layers[0].self_attn.q_proj
    pretrained = ordinary.weight.detach().clone()
    with actors.use_conditioning_actor("refined"):
        refined = ordinary.weight.detach().clone()

    assert not torch.equal(pretrained, refined)
    assert actors.generation_parameter_count == 1
    assert actors.conditioning_parameter_count == 2


def test_sensenova_actor_uses_precomputed_reference_embedding(tmp_path) -> None:
    _actor_export(tmp_path)
    raw = _ToyRawModel()
    actors = SenseNovaGenerationActors(raw, tmp_path)
    reference = SenseNovaReferenceState(
        embeddings=torch.tensor([[7.0]]),
        grid_hw=torch.tensor([[1, 1]]),
        width=32,
        height=32,
    )

    actors.begin("base", num_steps=1, reference_state=reference)
    encoded = raw.extract_feature(
        torch.tensor([[99.0]]),
        gen_model=False,
        grid_hw=torch.tensor([[1, 1]]),
    )
    raw.extract_feature(
        torch.tensor([[1.0]]),
        gen_model=True,
        grid_hw=torch.tensor([[1, 1]]),
    )

    assert encoded.item() == 7.0
    assert actors.finish() == ("base",)


def test_sensenova_continuous_reference_matches_official_normalization() -> None:
    class Raw(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        @staticmethod
        def patchify(images, patch_size, channel_first=False):
            assert channel_first
            batch, channels, height, width = images.shape
            return (
                images.reshape(
                    batch,
                    channels,
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                )
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(batch, -1, channels * patch_size**2)
            )

    raw = Raw()
    engine = SenseNovaPolicyInference.__new__(SenseNovaPolicyInference)
    engine.model = SimpleNamespace(runtime=SimpleNamespace(model=raw))
    engine.actors = SimpleNamespace(
        _original_extract_feature=lambda patches, **_kwargs: patches,
        use_conditioning_actor=lambda _actor: nullcontext(),
    )
    engine.reference_patch = 2
    pixels = torch.zeros((1, 3, 2, 2), dtype=torch.bfloat16)

    reference = engine.encode_reference_pixels(pixels)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.bfloat16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.bfloat16).view(1, 3, 1, 1)
    expected = raw.patchify((pixels * 0.5 + 0.5 - mean) / std, 2, True)[0]
    assert reference.embeddings.dtype == torch.bfloat16
    assert torch.equal(reference.embeddings, expected)
    assert torch.equal(reference.grid_hw, torch.tensor([[1, 1]]))


class _FakeEngine:
    def __init__(self) -> None:
        self.inputs = []
        self.image_multiple = 32

    def generate(self, source_image, *, prompt, seed, policy, config):
        del prompt, seed
        self.inputs.append((policy, source_image.getpixel((0, 0))[0]))
        increment = 1 if policy == "base" else 10
        value = min(255, source_image.getpixel((0, 0))[0] + increment)
        size = (
            int(config.get("width", source_image.size[0])),
            int(config.get("height", source_image.size[1])),
        )
        return (
            Image.new("RGB", size, (value, value, value)),
            tuple(
                "base" if policy == "base" else "refined"
                for _ in range(config["num_steps"])
            ),
        )


class _FakeNativeEngine:
    image_multiple = 32

    def __init__(self) -> None:
        self.references = []

    @staticmethod
    def conditioning_actor_for(policy, *, num_steps):
        assert policy.num_steps == num_steps
        return (
            "refined"
            if any(policy.actor_at(step) == "refined" for step in range(num_steps))
            else "base"
        )

    @staticmethod
    def encode_reference_image(_image, *, actor):
        assert actor in {"base", "refined"}
        return 0

    def generate_model_native(
        self,
        geometry,
        *,
        reference_state,
        prompt,
        seed,
        policy,
        config,
        encode_next_reference,
    ):
        del prompt, seed
        self.references.append((policy.name, reference_state))
        increment = 1 if policy.name == "base" else 10
        next_reference = reference_state + increment
        value = next_reference / 20.0
        width = int(config.get("width", geometry.size[0]))
        height = int(config.get("height", geometry.size[1]))
        token_count = (width // 32) * (height // 32)
        state = torch.full((1, token_count, 32 * 32 * 3), value)
        actor = "base" if policy.name == "base" else "refined"
        return (
            state,
            next_reference if encode_next_reference else None,
            (actor,) * config["num_steps"],
        )


@pytest.mark.parametrize("mode", ["legacy", "image", "model-native", "base"])
def test_sensenova_runner_saves_panels_once_and_resume_is_read_only(
    tmp_path, monkeypatch, mode
) -> None:
    from tailsteer.inference import sensenova_runner as runner

    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (512, 512), "black").save(image_root / "source.png")
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "example",
                        "image": "source.png",
                        "rounds": ["first", "second"],
                    }
                ]
            }
        )
    )
    output = tmp_path / "output"
    config = {
        "model": {},
        "run": {"output_dir": str(output), "policy_subdir": False},
        "schedule": runner.SENSENOVA_FIXED_SCHEDULE,
        "inference": {
            "tasks": str(tasks_path),
            "image_root": str(image_root),
            "export_path": str(tmp_path / "export"),
            "num_steps": 2,
            "feedback": "image" if mode in {"legacy", "base"} else mode,
        },
    }
    if mode != "legacy":
        config["inference"]["policies"] = {
            name: {"segments": [{"start": 0, "end": 2, "actor": actor}]}
            for name, actor in (("base", "base"), ("tailsteer", "refined"))
        }
    engine = _FakeNativeEngine() if mode == "model-native" else _FakeEngine()
    if mode in {"image", "base"}:
        generate = engine.generate

        def segmented_generate(source, *, policy, **kwargs):
            return generate(source, policy=policy.name, **kwargs)

        engine.generate = segmented_generate
    engine.actors = SimpleNamespace(
        manifest={"step": 2000},
        export=SimpleNamespace(schema_version=3),
        parameter_count=1,
        generation_parameter_count=1,
        conditioning_parameter_count=0,
    )
    if mode == "base":
        config["inference"].pop("export_path")
        config["inference"]["policies"].pop("tailsteer")
        engine.actors.manifest = {}
        engine.actors.export = None
    monkeypatch.setattr(runner, "_distributed_coordinates", lambda: (0, 1, 0))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(runner, "create_model", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "SenseNovaPolicyInference", lambda *a: engine)

    saved = set()
    save_image = Image.Image.save

    def save_once(image, path, *args, **kwargs):
        assert path not in saved, f"redundant image rewrite: {path}"
        saved.add(path)
        return save_image(image, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save_once)
    manifest = runner.run_sensenova_inference(config)
    if mode == "base":
        assert manifest["export_step"] is None
        assert manifest["export_schema_version"] is None
    assert (output / "COMPLETE").is_file()
    assert len(manifest["tasks"]) == 1
    task = manifest["tasks"][0]
    assert (output / task["source_comparison"]).is_file()
    assert all(
        (output / record["comparison_panel"]).is_file() for record in task["results"]
    )
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}

    def unexpected(*args, **kwargs):
        raise AssertionError("completed resume must not load models or render images")

    monkeypatch.setattr(runner, "create_model", unexpected)
    monkeypatch.setattr(Image.Image, "save", unexpected)
    assert runner.run_sensenova_inference(config, resume=True) == manifest
    assert {p: p.stat().st_mtime_ns for p in before} == before


def test_sensenova_multiround_policies_feed_their_own_outputs(tmp_path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (32, 32), "black").save(image_root / "source.png")
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "example",
                        "image": "source.png",
                        "rounds": ["first edit", "second edit"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    task = load_tasks(tasks_path, default_steps=2, default_seed=0)[0]
    engine = _FakeEngine()

    manifest = _run_task(
        task=task,
        engine=engine,
        inference={"num_steps": 2, "activation_step": 1},
        image_root=image_root,
        output=tmp_path / "output",
        refined_policy="late_step",
    )

    assert engine.inputs == [
        ("base", 0),
        ("late_step", 0),
        ("base", 1),
        ("late_step", 10),
    ]
    assert manifest["independent_feedback_chains"]
    assert len(manifest["results"]) == 2
    assert "metrics_pretrained_vs_input" in manifest["results"][0]
    assert "metrics_refined_vs_input" in manifest["results"][0]


def test_sensenova_model_native_feedback_saves_states_then_decodes(tmp_path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (32, 32), "black").save(image_root / "source.png")
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "example",
                        "image": "source.png",
                        "rounds": ["first edit", "second edit"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    task = load_tasks(tasks_path, default_steps=2, default_seed=0)[0]
    policies = {
        "base": InferencePolicy(
            name="base",
            segments=(PolicySegment(0, 2, "base"),),
            num_steps=2,
        ),
        "tailsteer": InferencePolicy(
            name="tailsteer",
            segments=(PolicySegment(0, 2, "refined"),),
            num_steps=2,
        ),
    }
    engine = _FakeNativeEngine()
    output = tmp_path / "output"

    manifest = _run_segmented_model_native_task(
        task=task,
        engine=engine,
        inference={"num_steps": 2},
        policies=policies,
        image_root=image_root,
        output=output,
    )

    assert engine.references == [
        ("base", 0),
        ("tailsteer", 0),
        ("base", 1),
        ("tailsteer", 10),
    ]
    assert manifest["feedback"] == "model_native"
    assert manifest["native_state"] == "z_final_pixels+reference_embedding"
    assert manifest["inter_round_feedback_quantized"] is False
    assert manifest["output_materialization"] == "all_rounds_after_chain"
    assert manifest["decoded_after_chain"] == "all_rounds"
    assert manifest["input"]["conditioning_actors"] == {
        "base": "base",
        "tailsteer": "refined",
    }
    assert len(manifest["results"]) == 4
    for record in manifest["results"]:
        assert (output / record["state"]).is_file()
        assert (output / record["image"]).is_file()


def test_sensenova_native_resolution_matches_official_reference_grid() -> None:
    assert sensenova_native_resolution((672, 384)) == (704, 416)
    assert sensenova_native_resolution((1024, 1024)) == (1024, 1024)
    assert sensenova_native_resolution((4096, 4096)) == (2048, 2048)


def test_sensenova_input_keeps_original_for_official_native_preprocessing() -> None:
    source = Image.new("RGB", (672, 384), "black")

    aligned, metadata = _prepare_model_input(
        source,
        inference={},
        image_multiple=32,
    )

    assert aligned.size == source.size
    assert metadata["reference_size"] == [704, 416]
    assert metadata["generation_size"] == [704, 416]
    assert metadata["token_grid"] == {"width": 22, "height": 13}
    assert metadata["policy"] == "official_native_no_crop"
    assert metadata["crop_fraction"] == 0.0


def test_sensenova_rejects_explicit_size_that_reference_would_resize() -> None:
    source = Image.new("RGB", (672, 384), "black")

    try:
        _prepare_model_input(
            source,
            inference={"width": 672, "height": 384},
            image_multiple=32,
        )
    except ValueError as error:
        assert "would be resized again" in str(error)
    else:
        raise AssertionError("expected non-native explicit resolution to fail")


def test_sensenova_balances_whole_chains_by_round_count() -> None:
    class Task:
        def __init__(self, rounds: int) -> None:
            self.rounds = tuple(range(rounds))

    tasks = tuple(Task(rounds) for rounds in (20, 20, 20, 10, 10, 10, 10, 10, 10))

    shards = balanced_task_indices(tasks, 4)
    assigned = sorted(index for shard in shards for index in shard)
    loads = [sum(len(tasks[index].rounds) for index in shard) for shard in shards]

    assert assigned == list(range(len(tasks)))
    assert loads == [30, 30, 30, 30]


class _FakeSenseNovaTraceModel(torch.nn.Module):
    patch_size = 2
    downsample_ratio = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.prompts = []

    @staticmethod
    def _apply_time_schedule(timesteps, _image_seq_len, timestep_shift):
        return timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)

    @staticmethod
    def patchify(pixels, patch):
        batch, channels, height, width = pixels.shape
        return (
            pixels.reshape(
                batch,
                channels,
                height // patch,
                patch,
                width // patch,
                patch,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, height // patch * width // patch, channels * patch * patch)
        )

    @staticmethod
    def unpatchify(state, patch, height, width):
        batch = state.shape[0]
        channels = state.shape[-1] // (patch * patch)
        return (
            state.reshape(
                batch,
                height // patch,
                width // patch,
                channels,
                patch,
                patch,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, height, width)
        )

    @staticmethod
    def _t2i_predict_v(_embeds, _indexes, _mask, _cache, _t, state, **_kwargs):
        return torch.ones_like(state)

    def it2i_generate(
        self,
        _tokenizer,
        prompt,
        images,
        *,
        image_size,
        num_steps,
        timestep_shift,
        **_kwargs,
    ):
        import numpy as np

        self.prompts.append(prompt)
        pixels = (
            torch.from_numpy(
                np.asarray(
                    images[0].convert("RGB").resize(image_size), dtype=np.float32
                ).copy()
                / 127.5
                - 1.0
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        state = self.patchify(pixels, 2)
        timesteps = self._apply_time_schedule(
            torch.linspace(0, 1, num_steps + 1), state.shape[1], timestep_shift
        )
        for index in range(num_steps):
            velocity = self._t2i_predict_v(
                None, None, None, None, timesteps[index], state
            )
            state = state + (timesteps[index + 1] - timesteps[index]) * velocity
        return self.unpatchify(state, 2, image_size[1], image_size[0])


def test_sensenova_official_trace_captures_forward_50_before_final() -> None:
    model = _FakeSenseNovaTraceModel()
    runtime = SenseNovaRuntime(
        model=model,
        tokenizer=object(),
        source_root=Path("/source"),
        model_root=Path("/model"),
        variant="sensenova-u1.5-8b-mot-preview",
        device="cpu",
    )

    trace = run_sensenova_edit(
        runtime,
        source_image=Image.new("RGB", (32, 32), (20, 40, 60)),
        prompt="edit",
        seed=7,
        width=32,
        height=32,
        num_inference_steps=50,
        capture_forward_steps=(42, 45, 47, 49, 50),
    )

    assert tuple(trace.forward_states) == (42, 45, 47, 49, 50)
    assert trace.source_state.shape == trace.final_state.shape
    assert not torch.equal(trace.forward_states[50], trace.final_state)
    assert trace.image.size == (32, 32)
    assert len(trace.timesteps) == 51
    assert model.prompts == ["<image>\nedit"]
