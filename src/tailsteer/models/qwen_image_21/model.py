"""Qwen-Image 2.1 velocities for the shared final A/B/C/D recipes.

DiT-only by default; optionally train Qwen3-VL as well. Recompute the causal
prefix on supervised forwards so generation KV cannot truncate its gradients.
"""

from __future__ import annotations

from copy import deepcopy

from tailsteer.models.native_transformer import (
    NativeTransformerTailModel,
    TransformerConditioning,
)
from tailsteer.models.native_transformer.model import fit_source_image, unwrap
from tailsteer.models.protocols import TailVelocityBranches, TrajectoryPoint

from .official import QWEN21_SOURCE_REVISION, QwenImage21Runtime, model_load_slot


class QwenImage21Model(NativeTransformerTailModel):
    export_format = "tailsteer_qwen_image_21_transformer"
    source_revision = QWEN21_SOURCE_REVISION
    joint_training = False

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
            raise ValueError("Qwen exports are loaded by its inference runtime")
        if float(config.get("true_cfg_scale", 1.0)) != 1.0:
            raise ValueError("Qwen training matches guidance-free cached trajectories")
        device = torch.device(device or f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
        with torch.random.fork_rng(devices=[device.index or 0]):
            runtime = QwenImage21Runtime.load(config, device=device)
        pipe = runtime.pipeline
        result = cls(
            refined_transformer=pipe.transformer,
            pretrained_transformer=pipe.transformer,
            device=device,
            dtype=torch.bfloat16,
            model_name="qwen_image_21",
            model_path=config["path"],
            layer_checkpointing=bool(gradient_checkpointing),
        )
        result.pipe = pipe
        result.joint_training = False
        result._independent_actors = bool(independent_actors)
        result._teacher_pending = bool(independent_actors)
        if not include_vae:
            pipe.vae = None
        # A large teacher is loaded only after the student has been sharded.
        if independent_actors and int(os.environ.get("WORLD_SIZE", "1")) == 1:
            result._load_teacher()
        return result

    def configure_trainable(self, config):
        modules = list(config.get("additional_modules", []))
        if not modules:
            return super().configure_trainable(config)
        if modules != ["text_encoder"]:
            raise ValueError("Qwen additional_modules only supports text_encoder")
        if not config.get("conditioning_gradients", True):
            raise ValueError("Qwen joint training requires conditioning_gradients")
        from .joint import ConditionEncoder, JointDenoiser

        super().configure_trainable({**config, "additional_modules": []})
        encoder = ConditionEncoder(self.pipe.text_encoder.model)
        encoder.requires_grad_(True)
        # Official conditioning deliberately bypasses this final RMSNorm.
        encoder.model.language_model.norm.requires_grad_(False)
        self.refined_transformer = JointDenoiser(
            self.refined_transformer, encoder, self.pipe
        ).train()
        self.joint_training = True
        self.trainable_config = deepcopy(dict(config))
        self.trainable_config["conditioning_gradients"] = True
        self._replace_pipeline_transformer(self.refined_transformer)
        # Include the tiny frozen final norm so the portable component has a
        # complete parameter set; it is never added to the optimizer.
        self._export_names = frozenset(
            dict(self.refined_transformer.named_parameters())
        )
        if not self._independent_actors:
            self.pretrained_transformer = self.refined_transformer
        elif not self._teacher_pending:
            self._load_teacher_encoder()

    def _load_teacher_encoder(self):
        import logging

        import torch
        from transformers import Qwen3VLForConditionalGeneration

        from .joint import ConditionEncoder, JointDenoiser

        logger = logging.getLogger("tailsteer")
        logger.info("Qwen joint teacher: loading independent frozen Qwen3-VL")
        with torch.random.fork_rng(devices=[self.device.index or 0]), model_load_slot():
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path / "text_encoder",
                dtype=self.dtype,
                local_files_only=True,
            )
            encoder = ConditionEncoder(encoder.model).to(self.device)
        self.pretrained_transformer = (
            JointDenoiser(self.pretrained_transformer, encoder, self.pipe)
            .requires_grad_(False)
            .eval()
        )
        logger.info("Qwen joint teacher: Qwen3-VL loaded; ready for FSDP sharding")

    def _load_teacher(self):
        import torch
        from diffusers import QwenImage21Transformer2DModel

        if not self._teacher_pending:
            return
        with torch.random.fork_rng(devices=[self.device.index or 0]), model_load_slot():
            self.pretrained_transformer = QwenImage21Transformer2DModel.from_pretrained(
                self.model_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
                local_files_only=True,
            ).to(self.device)
        self.pretrained_transformer.requires_grad_(False).eval()
        self._teacher_pending = False
        if self.joint_training:
            self._load_teacher_encoder()

    def auxiliary_sharding_groups(self):
        self._load_teacher()
        return super().auxiliary_sharding_groups()

    def _replace_pipeline_transformer(self, module):
        if self.joint_training:
            root = unwrap(module)
            self.pipe.transformer = root.transformer
            self.pipe.text_encoder = root.text_encoder
        else:
            self.pipe.transformer = module

    @staticmethod
    def inference_export_dtype():
        import torch

        return torch.bfloat16

    def transformer_blocks(self, transformer):
        if self.joint_training:
            return transformer.sharding_units()
        return tuple(transformer.transformer_blocks)

    @staticmethod
    def defer_activation_checkpointing():
        return True

    def _checkpoint_locations(self):
        root = unwrap(self.refined_transformer)
        if self.joint_training:
            return root.block_locations()
        return tuple(
            (root, "transformer_blocks", i, layer)
            for i, layer in enumerate(root.transformer_blocks)
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

        if len(prompts) != 1 or (height, width) != (1024, 1024):
            raise ValueError(
                "Qwen training currently uses batch_size=1, 1024-square cache"
            )
        if tuple(source_state.shape) != (1, 64, 64, 64):
            raise ValueError(
                "Qwen cache must contain normalized [B,64,H/16,W/16] states"
            )
        image = fit_source_image(metadata[0]["source_path"], width=width, height=height)
        image = self.pipe.image_processor.resize(
            image.convert("RGBA"), width=width, height=height
        )
        if self.joint_training:
            return TransformerConditioning(
                payload={
                    "source": source_state.flatten(2).transpose(1, 2).detach(),
                    "image": image,
                    "prompts": list(prompts),
                    "img_shapes": [[(1, height // 16, width // 16)] * 2],
                    "encoded": {},
                },
                batch_size=1,
            )
        with torch.no_grad():
            embeds, mask, image_mask = self.pipe.encode_prompt(
                image=[image],
                prompt=list(prompts),
                device=self.device,
                num_images_per_prompt=1,
            )
        tokens = source_state.flatten(2).transpose(1, 2).detach()
        return TransformerConditioning(
            payload={
                "source": tokens,
                "encoder_hidden_states": embeds.detach(),
                "encoder_hidden_states_mask": mask.detach()
                if mask is not None
                else None,
                "img_mask": torch.cat(
                    (image_mask, image_mask.new_ones(1, tokens.shape[1] // 4)),
                    dim=1,
                ),
                "img_shapes": [[(1, height // 16, width // 16)] * 2],
            },
            batch_size=1,
        )

    @staticmethod
    def tail_velocity_branch_names():
        return ("conditioned",)

    def predict_tail_velocity_branches(self, latent, point, conditioning, *, actor):
        import torch

        if actor == "base" and self._teacher_pending:
            raise RuntimeError("Qwen teacher must be prepared after student sharding")
        if conditioning.batch_size != latent.shape[0]:
            raise ValueError("Qwen conditioning and state batch differ")
        state = latent.to(device=self.device, dtype=self.dtype)
        tokens = state.flatten(2).transpose(1, 2)
        values = dict(conditioning.payload)
        hidden = torch.cat((values.pop("source"), tokens), dim=1)
        # Match the pipeline's BF16 cast BEFORE division by 1000.
        timestep = state.new_full((state.shape[0],), float(point.timestep)) / 1000
        transformer = self.actor_transformer(actor)
        with self.autocast_context(transformer):
            kwargs = (
                {"conditioning": conditioning.payload, "actor": actor}
                if self.joint_training
                else {
                    **values,
                    "kv_cache": None,
                    "kv_cache_mode": None,
                    "return_dict": False,
                }
            )
            prediction = transformer(
                hidden_states=hidden,
                timestep=timestep,
                **kwargs,
            )[0][:, -tokens.shape[1] :]
        velocity = prediction.transpose(1, 2).reshape_as(state)
        return TailVelocityBranches({"conditioned": velocity})

    @staticmethod
    def compose_tail_velocity_branches(prediction):
        if set(prediction.values) != {"conditioned"}:
            raise ValueError("Qwen base cache has one conditioned branch")
        return prediction.values["conditioned"]

    def tail_schedule(self, *, num_steps, height, width):
        import numpy as np
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
            calculate_shift,
            retrieve_timesteps,
        )

        if num_steps != 40 or (height, width) != (1024, 1024):
            raise ValueError(
                "Qwen training matches the official 40-step 1024-square cache"
            )
        scheduler = type(self.pipe.scheduler).from_config(self.pipe.scheduler.config)
        cfg = scheduler.config
        mu = calculate_shift(
            height * width // 256,
            cfg.get("base_image_seq_len", 256),
            cfg.get("max_image_seq_len", 4096),
            cfg.get("base_shift", 0.5),
            cfg.get("max_shift", 1.15),
        )
        retrieve_timesteps(
            scheduler,
            num_steps,
            "cpu",
            sigmas=np.linspace(1.0, 1 / num_steps, num_steps),
            mu=mu,
        )
        return {
            k: TrajectoryPoint(
                float(k),
                float(sigma),
                float(scheduler.timesteps[k]) if k < num_steps else 0.0,
            )
            for k, sigma in enumerate(scheduler.sigmas)
        }

    @staticmethod
    def tail_terminal_delta(point):
        return -float(point.sigma)

    @staticmethod
    def tail_step_delta(current, following):
        return float(following.sigma) - float(current.sigma)

    @staticmethod
    def tail_advance(latent, velocity, *, current, following):
        import torch

        # Official Euler: FP32 accumulation, then cast back to velocity dtype.
        # FlowMatchEulerDiscreteScheduler deliberately stores sigmas on CPU.
        # A CUDA 0-d scalar changes BF16 multiplication rounding (even though
        # its value/dtype match), so preserve the native CPU-scalar path.
        sigma = torch.tensor(current.sigma, dtype=torch.float32)
        following_sigma = sigma.new_tensor(following.sigma)
        return (latent.float() + (following_sigma - sigma) * velocity).to(
            velocity.dtype
        )

    @staticmethod
    def tail_keep_mask(mask, state):
        if (
            mask.ndim != 4
            or mask.shape[0] != state.shape[0]
            or mask.shape[1] not in (1, state.shape[1])
            or mask.shape[2:] != state.shape[2:]
        ):
            raise ValueError("Qwen mask must use [B,1|64,H/16,W/16] layout")
        return mask.to(device=state.device, dtype=state.dtype)

    def _release_runtime(self):
        self.pipe = None

    def prepare_inference_export(self, state, *, step):
        selected, filename, manifest = super().prepare_inference_export(
            state, step=step
        )
        if self.joint_training:
            from .joint import JOINT_EXPORT_FORMAT

            filename = "model.safetensors"
            manifest.update(
                format=JOINT_EXPORT_FORMAT,
                weights=filename,
                components=["transformer", "text_encoder"],
            )
        return selected, filename, manifest
