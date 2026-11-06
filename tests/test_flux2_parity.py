from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import Flux2Transformer2DModel  # noqa: E402

from tailsteer.models.flux2.model import (  # noqa: E402
    Flux2Conditioning,
    Flux2Model,
    SchedulePoint,
    ScheduleTransition,
    build_flux2_schedule,
)
from tailsteer.models.protocols import TailVelocityBranches  # noqa: E402


def test_flux2_schedule_exactly_matches_completed_v2_cache_contract() -> None:
    expected = {
        0: (1.0, 1000.0),
        23: (0.898775577545166, 898.7755737304688),
        33: (0.7957689166069031, 795.7689208984375),
        38: (0.7048854231834412, 704.8854370117188),
        42: (0.5902800559997559, 590.280029296875),
        43: (0.5518284440040588, 551.8284301757812),
        44: (0.5077296495437622, 507.7296447753906),
        45: (0.4566407799720764, 456.6407775878906),
        46: (0.3967570662498474, 396.757080078125),
        47: (0.32559332251548767, 325.59332275390625),
        48: (0.23963116109371185, 239.63116455078125),
        49: (0.13371895253658295, 133.7189483642578),
    }
    schedule = build_flux2_schedule(num_steps=50, height=1024, width=1024)

    assert {
        step: (schedule.point(step).sigma, schedule.point(step).timestep)
        for step in expected
    } == expected


def test_flux2_v2_euler_update_stays_in_native_bf16() -> None:
    latent = torch.randn((1, 8, 4, 4), dtype=torch.bfloat16)
    velocity = torch.randn_like(latent)
    current = SchedulePoint(49, 0.13371895253658295, 133.7189483642578)
    transition = ScheduleTransition(current, 0.0)

    actual = Flux2Model.advance(latent, velocity, transition)
    expected = latent + velocity * (transition.next_sigma - current.sigma)
    tail_actual = Flux2Model.tail_advance(
        latent,
        velocity,
        current=current,
        following=SchedulePoint(50, 0.0, 0.0),
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(tail_actual, expected, rtol=0, atol=0)


class _Pipe:
    text_encoder = None
    vae = None

    def __init__(self, transformer):
        self.transformer = transformer
        self.config = SimpleNamespace(is_distilled=False)
        self.prompt = torch.randn((1, 3, 16))
        self.text_ids = torch.zeros((1, 3, 4))

    @staticmethod
    def _pack_latents(latent):
        return latent.permute(0, 2, 3, 1).reshape(latent.shape[0], -1, latent.shape[1])

    def encode_prompt(self, *, prompt, device, num_images_per_prompt):
        del num_images_per_prompt
        batch = len(prompt)
        return (
            self.prompt.to(device).expand(batch, -1, -1),
            self.text_ids.to(device).expand(batch, -1, -1),
        )

    @staticmethod
    def _prepare_latent_ids(latent):
        return torch.zeros((latent.shape[0], latent.shape[-2] * latent.shape[-1], 4))

    @staticmethod
    def _prepare_image_ids(images):
        latent = images[0]
        return torch.ones((latent.shape[0], latent.shape[-2] * latent.shape[-1], 4))


def test_tailsteer_disabled_path_matches_upstream_transformer_exactly() -> None:
    torch.manual_seed(7)
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=16,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    ).eval()
    pipe = _Pipe(transformer)
    model = Flux2Model(
        pipe=pipe,
        refined_transformer=transformer,
        pretrained_transformer=transformer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg_scale=1.0,
        negative_prompt="",
        cfg_training="positive_only",
    )
    latent = torch.randn((1, 8, 2, 2))
    source_tokens = torch.randn((1, 4, 8))
    target_ids = torch.zeros((1, 4, 4))
    source_ids = torch.ones((1, 4, 4))
    prompt = torch.randn((1, 3, 16))
    text_ids = torch.zeros((1, 3, 4))
    conditioning = Flux2Conditioning(
        {
            "source_tokens": source_tokens,
            "target_ids": target_ids,
            "source_ids": source_ids,
            "positive": prompt,
            "positive_ids": text_ids,
            "negative": None,
            "negative_ids": None,
        },
        batch_size=1,
    )
    point = SchedulePoint(0, 0.5, 500.0)

    actual = model.predict_velocity(latent, point, conditioning, actor="base")
    target_tokens = pipe._pack_latents(latent)
    direct = transformer(
        hidden_states=torch.cat([target_tokens, source_tokens], dim=1),
        timestep=torch.tensor([0.5]),
        guidance=None,
        encoder_hidden_states=prompt,
        txt_ids=text_ids,
        img_ids=torch.cat([target_ids, source_ids], dim=1),
        joint_attention_kwargs=None,
        return_dict=False,
    )[0][:, :4]
    direct = direct.reshape(1, 2, 2, 8).permute(0, 3, 1, 2)
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)

    cached_batch = {
        "latent": latent,
        "source_latent": torch.randn_like(latent),
        "keep_mask": torch.ones((1, 1, 2, 2)),
        "metadata": [
            {
                "prompt": "cached state",
                "height": 32,
                "width": 32,
                "step_index": 49,
                "sigma": 0.02,
                "timestep": 20.0,
            }
        ],
    }
    cached_prediction = model.training_forward(cached_batch)
    assert cached_prediction.shape == latent.shape
    assert torch.isfinite(cached_prediction).all()


def test_flux2_branch_composition_matches_guided_velocity() -> None:
    model = Flux2Model.__new__(Flux2Model)
    model.pipe = SimpleNamespace(config=SimpleNamespace(is_distilled=False))
    model.cfg_training = "guided"
    model.cfg_scale = 4.0
    base = torch.randn((2, 4, 3, 3))
    conditioned = torch.randn_like(base)

    actual = model.compose_tail_velocity_branches(
        TailVelocityBranches({"base": base, "conditioned": conditioned})
    )

    torch.testing.assert_close(actual, base + 4.0 * (conditioned - base))
    assert model.tail_velocity_branch_names() == ("base", "conditioned")
