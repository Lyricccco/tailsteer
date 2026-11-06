"""Optional DiT + Qwen3-VL conditioning backbone, using the official encoder path.

The raw conditioning is batch-local. Only no-grad rollout/teacher embeddings
are memoized; a supervised call always rebuilds the student's autograd graph.
"""

from copy import copy

import torch

JOINT_EXPORT_FORMAT = "tailsteer_qwen_image_21_joint"


class ConditionEncoder(torch.nn.Module):
    """Expose Qwen3-VL hidden states without computing the unused LM logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, **kwargs):
        # The official encode_prompt installs its pre-final-norm hook and
        # extracts hidden_states[-1]. Keep that exact representation.
        return self.model(**{**kwargs, "use_cache": False})


def encode_conditioning(pipe, values):
    embeds, mask, image_mask = pipe.encode_prompt(
        image=[values["image"]],
        prompt=values["prompts"],
        device=values["source"].device,
        num_images_per_prompt=1,
    )
    return {
        "encoder_hidden_states": embeds,
        "encoder_hidden_states_mask": mask,
        "img_mask": torch.cat(
            (image_mask, image_mask.new_ones(1, values["source"].shape[1] // 4)),
            dim=1,
        ),
        "img_shapes": values["img_shapes"],
    }


class JointDenoiser(torch.nn.Module):
    """One FSDP/checkpoint root; external VAE is deliberately not a child."""

    def __init__(self, transformer, text_encoder, pipeline):
        super().__init__()
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.pipeline = copy(pipeline)
        self.pipeline.transformer = transformer
        self.pipeline.text_encoder = text_encoder
        self.pipeline.vae = None

    def forward(self, *, hidden_states, timestep, conditioning, actor):
        cached = conditioning["encoded"]
        if torch.is_grad_enabled():
            values = encode_conditioning(self.pipeline, conditioning)
        else:
            if actor not in cached:
                cached[actor] = encode_conditioning(self.pipeline, conditioning)
            values = cached[actor]
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            **values,
            kv_cache=None,
            kv_cache_mode=None,
            return_dict=False,
        )

    def block_locations(self):
        core = self.text_encoder.model
        return (
            *((core.visual, "blocks", i, b) for i, b in enumerate(core.visual.blocks)),
            *(
                (core.language_model, "layers", i, b)
                for i, b in enumerate(core.language_model.layers)
            ),
            *(
                (self.transformer, "transformer_blocks", i, b)
                for i, b in enumerate(self.transformer.transformer_blocks)
            ),
        )

    def sharding_units(self):
        # Children before parents: embeddings/mergers and DiT projections also
        # get their own forward-triggered all-gathers, not a 30-GB root gather.
        locations = self.block_locations()
        return (
            *(item[3] for item in locations if item[0] is not self.transformer),
            self.text_encoder,
            *self.transformer.transformer_blocks,
            self.transformer,
        )
