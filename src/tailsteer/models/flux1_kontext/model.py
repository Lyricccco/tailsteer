"""Differentiable BFL Kontext adapter for the shared A/B/C/D algorithms."""

from __future__ import annotations

from pathlib import Path

from tailsteer.models.native_transformer import (
    NativeTransformerTailModel,
    TransformerConditioning,
)
from tailsteer.models.native_transformer.model import unwrap
from tailsteer.models.protocols import TailVelocityBranches, TrajectoryPoint

from .official import (
    KONTEXT_SOURCE_REVISION,
    _load_component,
    _load_embedder,
    import_source,
)


class KontextModel(NativeTransformerTailModel):
    export_format = "tailsteer_kontext_transformer"
    source_revision = KONTEXT_SOURCE_REVISION
    branch_names = ("conditioned",)

    def __init__(self, *, t5, clip, guidance_scale=2.5, **kwargs):
        super().__init__(**kwargs)
        self.t5, self.clip = t5, clip
        self.guidance_scale = float(guidance_scale)

    @classmethod
    def load(
        cls,
        config,
        *,
        device=None,
        include_vae=False,
        independent_actors=False,
        gradient_checkpointing=False,
        export_path=None,
    ):
        import os

        import torch

        if export_path is not None:
            raise ValueError("Kontext exports are loaded by the inference runner")
        if include_vae:
            raise ValueError(
                "Kontext uses post-checkpoint inference; "
                "keep visual_validation disabled"
            )
        if config.get("dtype", "bfloat16") != "bfloat16":
            raise ValueError("Kontext uses the native BF16 compute configuration")
        import_source(config["code_path"])
        from flux.model import Flux
        from flux.util import configs

        device = torch.device(
            device or f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        )
        root = Path(config["path"]).resolve()

        def load_transformer():
            return _load_component(
                Flux,
                configs["flux-dev-kontext"].params,
                root / "flux1-kontext-dev.safetensors",
                device,
                dtype=torch.bfloat16,
            )

        refined = load_transformer()
        teacher = load_transformer() if independent_actors else refined
        return cls(
            refined_transformer=refined,
            pretrained_transformer=teacher,
            device=device,
            dtype=torch.bfloat16,
            model_name="flux1_kontext_dev",
            model_path=root,
            layer_checkpointing=bool(config.get("layer_checkpointing", True)),
            t5=_load_embedder(root, is_clip=False, device=device),
            clip=_load_embedder(root, is_clip=True, device=device),
            guidance_scale=config.get("guidance_scale", 2.5),
        )

    def _replace_pipeline_transformer(self, module):
        # No pipeline owns another reference to the training transformer.
        pass

    def transformer_blocks(self, transformer):
        return tuple(transformer.double_blocks) + tuple(transformer.single_blocks)

    @staticmethod
    def defer_activation_checkpointing():
        # Recomputation must re-enter each FSDP block's unshard hook.
        return True

    def _checkpoint_locations(self):
        root = unwrap(self.refined_transformer)
        return tuple(
            (root, attribute, index, layer)
            for attribute in ("double_blocks", "single_blocks")
            for index, layer in enumerate(getattr(root, attribute))
        )

    @staticmethod
    def state_rank():
        return 4

    def prepare_conditioning_from_metadata(
        self,
        *,
        source_state,
        metadata,
        prompts,
        height,
        width,
    ):
        import torch
        from flux.sampling import prepare

        if tuple(source_state.shape) != (len(prompts), 16, height // 8, width // 8):
            raise ValueError("Kontext source cache must be an unpacked VAE latent")
        # Native prepare supplies T5/CLIP embeddings and FP32 position IDs.
        # Source/target share a spatial grid; only their modality ID differs.
        with torch.no_grad():
            payload = prepare(self.t5, self.clip, source_state, list(prompts))
            payload["img_cond_seq"] = payload.pop("img")
            source_ids = payload["img_ids"].clone()
            source_ids[..., 0] = 1
            payload["img_cond_seq_ids"] = source_ids
        return TransformerConditioning(payload=payload, batch_size=len(prompts))

    def tail_velocity_branch_names(self):
        return self.branch_names

    def predict_tail_velocity_branches(self, latent, point, conditioning, *, actor):
        import torch
        from einops import rearrange
        from flux.sampling import unpack

        if conditioning.batch_size != latent.shape[0]:
            raise ValueError("Kontext conditioning and state batch differ")
        transformer = self.actor_transformer(actor)
        device = next(unwrap(transformer).parameters()).device
        state = latent.to(device=device, dtype=self.dtype)
        tokens = rearrange(state, "b c (h p) (w q) -> b (h w) (c p q)", p=2, q=2)
        payload = {
            name: value.to(device) for name, value in conditioning.payload.items()
        }
        timestep = torch.full(
            (state.shape[0],), float(point.timestep), device=device, dtype=self.dtype
        )
        guidance = torch.full_like(timestep, self.guidance_scale)
        with self.autocast_context(transformer):
            prediction = transformer(
                img=torch.cat((tokens, payload["img_cond_seq"]), dim=1),
                img_ids=torch.cat(
                    (payload["img_ids"], payload["img_cond_seq_ids"]), dim=1
                ),
                txt=payload["txt"],
                txt_ids=payload["txt_ids"],
                y=payload["vec"],
                timesteps=timestep,
                guidance=guidance,
            )[:, : tokens.shape[1]]
        # Unlike the cache capture helper, native unpack preserves autograd.
        velocity = unpack(prediction, state.shape[-2] * 8, state.shape[-1] * 8)
        return TailVelocityBranches({"conditioned": velocity})

    def compose_tail_velocity_branches(self, prediction):
        if set(prediction.values) != set(self.branch_names):
            raise ValueError("Kontext has one guidance-distilled conditioned branch")
        return prediction.values["conditioned"]

    def tail_schedule(self, *, num_steps, height, width):
        import torch
        from flux.sampling import get_schedule

        if height % 16 or width % 16:
            raise ValueError("Kontext dimensions must be divisible by 16")
        sigmas = get_schedule(num_steps, height * width // 256, shift=True)
        return {
            step: TrajectoryPoint(
                float(step), float(sigma), float(torch.tensor(sigma, dtype=self.dtype))
            )
            for step, sigma in enumerate(sigmas)
        }

    @staticmethod
    def tail_step_delta(current, following):
        return float(following.sigma) - float(current.sigma)

    @staticmethod
    def tail_terminal_delta(point):
        return -float(point.sigma)

    @staticmethod
    def tail_advance(latent, velocity, *, current, following):
        # Match BFL denoise exactly, including native BF16 Euler arithmetic.
        return latent + (float(following.sigma) - float(current.sigma)) * velocity

    @staticmethod
    def tail_keep_mask(mask, state):
        if tuple(mask.shape) != (state.shape[0], 1, *state.shape[-2:]):
            raise ValueError("Kontext keep mask must use [B,1,H,W] latent layout")
        return mask.to(device=state.device, dtype=state.dtype)

    def _release_runtime(self):
        self.t5 = self.clip = None
