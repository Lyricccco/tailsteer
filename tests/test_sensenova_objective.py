from __future__ import annotations

import torch

from tailsteer.models.protocols import TailVelocityBranches
from tailsteer.models.sensenova.conditioning import (
    ImagePrefixKVAdapter,
    PrefixKVAdapterConfig,
)
from tailsteer.models.sensenova.model import SenseNovaModel
from tailsteer.models.sensenova.objective import sensenova_preservation_velocity
from tailsteer.models.sensenova.training import configure_sensenova_trainable


class _ToyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2, bias=False)
        self.q_proj_mot_gen = torch.nn.Linear(2, 2, bias=False)


class _ToyLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 2)
        self.model.norm_mot_gen = torch.nn.LayerNorm(2)
        self.model.attention = _ToyAttention()


class _ToySenseNova(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _ToyLanguageModel()
        self.vision_model = torch.nn.Linear(2, 2)
        self.fm_modules = torch.nn.ModuleDict(
            {
                "vision_model_mot_gen": torch.nn.Linear(2, 2),
                "timestep_embedder": torch.nn.Linear(2, 2),
                "noise_scale_embedder": torch.nn.Linear(2, 2),
                "fm_head": torch.nn.Linear(2, 2),
            }
        )


def test_generation_denoiser_excludes_ordinary_llm_and_source_vision() -> None:
    model = _ToySenseNova()

    selected = configure_sensenova_trainable(model, "generation_denoiser")

    assert selected
    assert all("mot_gen" in name or name.startswith("fm_modules.") for name in selected)
    assert not model.language_model.model.attention.q_proj.weight.requires_grad
    assert model.language_model.model.attention.q_proj_mot_gen.weight.requires_grad
    assert not model.language_model.model.embed_tokens.weight.requires_grad
    assert not model.vision_model.weight.requires_grad
    assert all(parameter.requires_grad for parameter in model.fm_modules.parameters())


def test_output_head_policy_only_trains_sensenova_fm_head() -> None:
    model = _ToySenseNova()

    selected = configure_sensenova_trainable(model, {"policy": "output_head"})

    assert selected
    assert all(name.startswith("fm_modules.fm_head.") for name in selected)
    assert all(
        parameter.requires_grad for parameter in model.fm_modules.fm_head.parameters()
    )
    assert not model.fm_modules.vision_model_mot_gen.weight.requires_grad
    assert not model.fm_modules.timestep_embedder.weight.requires_grad
    assert not model.language_model.model.norm_mot_gen.weight.requires_grad
    assert not model.vision_model.weight.requires_grad


def test_output_head_norm_policy_matches_flux_h1_tail() -> None:
    model = _ToySenseNova()

    selected = configure_sensenova_trainable(model, {"policy": "h1"})

    assert selected
    assert any(name.startswith("fm_modules.fm_head.") for name in selected)
    assert any(
        name.startswith("language_model.model.norm_mot_gen.") for name in selected
    )
    assert all(
        parameter.requires_grad for parameter in model.fm_modules.fm_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.language_model.model.norm_mot_gen.parameters()
    )
    assert not model.fm_modules.vision_model_mot_gen.weight.requires_grad
    assert not model.fm_modules.timestep_embedder.weight.requires_grad
    assert not model.language_model.model.attention.q_proj_mot_gen.weight.requires_grad
    assert model._tailsteer_trainable_groups == ("terminal_norm", "output_head")


def test_full_model_selects_every_u1_sft_parameter_and_conditioning_graph() -> None:
    model = _ToySenseNova()

    selected = set(configure_sensenova_trainable(model, {"policy": "full_model"}))

    assert selected == {name for name, _ in model.named_parameters()}
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model._tailsteer_conditioning_gradients is True


def test_conditioning_ablation_unfreezes_reference_and_attaches_adapter() -> None:
    model = _ToySenseNova()
    model.language_model.model.layers = torch.nn.ModuleList(
        [
            torch.nn.ModuleDict(
                {
                    "self_attn": torch.nn.Module(),
                }
            )
            for _ in range(2)
        ]
    )
    for layer in model.language_model.model.layers:
        layer.self_attn.head_dim = 2
        layer.self_attn.config = type("Config", (), {"num_key_value_heads": 1})()
        layer.self_attn.q_proj = torch.nn.Linear(2, 2, bias=False)

    selected = configure_sensenova_trainable(
        model,
        {
            "policy": "generation_denoiser",
            "reference_vision_model": True,
            "prefix_kv_adapter": {
                "enabled": True,
                "target": "image_tokens",
                "last_n_layers": 1,
                "rank": 1,
                "scale": 1.0,
            },
        },
    )

    assert model.vision_model.weight.requires_grad
    assert hasattr(model, "tailsteer_prefix_kv_adapter")
    assert any(name.startswith("vision_model.") for name in selected)
    assert any(name.startswith("tailsteer_prefix_kv_adapter.") for name in selected)
    assert not model.language_model.model.attention.q_proj.weight.requires_grad


def test_image_prefix_adapter_is_exactly_zero_initialized_and_image_only() -> None:
    from types import SimpleNamespace

    adapter = ImagePrefixKVAdapter(
        layer_count=2,
        head_dim=4,
        reference_width=6,
        head_count=2,
        config=PrefixKVAdapterConfig(enabled=True, last_n_layers=1, rank=2, scale=1.0),
    )
    first = SimpleNamespace(
        keys=torch.randn(1, 2, 5, 4), values=torch.randn(1, 2, 5, 4)
    )
    second = SimpleNamespace(
        keys=torch.randn(1, 2, 5, 4), values=torch.randn(1, 2, 5, 4)
    )
    cache = SimpleNamespace(layers=[first, second])
    before_first = (first.keys.clone(), first.values.clone())
    before_second = (second.keys.clone(), second.values.clone())

    reference = torch.randn(2, 6, requires_grad=True)
    adapter.apply(
        cache,
        torch.tensor([False, True, True, False, False]),
        reference,
    )

    torch.testing.assert_close(first.keys, before_first[0], rtol=0, atol=0)
    torch.testing.assert_close(first.values, before_first[1], rtol=0, atol=0)
    torch.testing.assert_close(second.keys, before_second[0], rtol=0, atol=0)
    torch.testing.assert_close(second.values, before_second[1], rtol=0, atol=0)
    loss = second.keys.square().mean() + second.values.square().mean()
    loss.backward()
    up_gradients = [
        parameter.grad
        for name, parameter in adapter.named_parameters()
        if name.endswith("up.weight")
    ]
    assert up_gradients and all(gradient is not None for gradient in up_gradients)

    # Zero-output initialization intentionally blocks the vision gradient on
    # update zero. Once the output projection moves, the explicit reference
    # branch carries a live task gradient into the complete vision encoder.
    for name, parameter in adapter.named_parameters():
        if name.endswith("up.weight"):
            torch.nn.init.constant_(parameter, 0.1)
    next_layer = SimpleNamespace(
        keys=torch.randn(1, 2, 5, 4), values=torch.randn(1, 2, 5, 4)
    )
    next_cache = SimpleNamespace(layers=[first, next_layer])
    next_reference = torch.randn(2, 6, requires_grad=True)
    adapter.apply(
        next_cache,
        torch.tensor([False, True, True, False, False]),
        next_reference,
    )
    (next_layer.keys.square().mean() + next_layer.values.square().mean()).backward()
    assert next_reference.grad is not None
    assert bool(next_reference.grad.abs().sum() > 0)


def test_sensenova_objective_matches_legacy_positive_time_formula() -> None:
    current = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
    source = torch.flip(current, dims=(1,))
    prediction = current * 0.1
    mask = torch.tensor([[[[1, 0], [1, 1]]], [[[0, 1], [1, 1]]]], dtype=torch.float32)
    sigma = 0.05769222974777222

    actual = sensenova_preservation_velocity(
        prediction, current, source, mask, sigma=sigma
    )

    flat_mask = mask[:, 0].reshape(2, 4, 1)
    target = (source - current) / sigma
    per_token = (prediction - target).square().mean(dim=-1, keepdim=True)
    expected = (per_token * flat_mask).sum(dim=(1, 2)) / flat_mask.sum(
        dim=(1, 2)
    ).clamp_min(1)
    torch.testing.assert_close(actual.per_sample, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.mean(), rtol=0, atol=0)


def test_sensenova_velocity_sign_is_intentionally_not_flux_sign() -> None:
    current = torch.ones((1, 4, 3))
    source = torch.zeros_like(current)
    mask = torch.ones((1, 1, 2, 2))
    sigma = 0.25
    positive_time_target = (source - current) / sigma

    result = sensenova_preservation_velocity(
        positive_time_target, current, source, mask, sigma=sigma
    )

    assert result.loss.item() == 0.0


def test_sensenova_branch_composition_matches_current_image_text_cfg() -> None:
    model = SenseNovaModel.__new__(SenseNovaModel)
    model.cfg_scale = 4.0
    image_conditioned = torch.randn((2, 4, 3))
    conditioned = torch.randn_like(image_conditioned)

    actual = model.compose_tail_velocity_branches(
        TailVelocityBranches({"base": image_conditioned, "conditioned": conditioned})
    )

    torch.testing.assert_close(
        actual, image_conditioned + 4.0 * (conditioned - image_conditioned)
    )
