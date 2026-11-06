"""Lightweight FLUX.2 VAE-only decoder for saved latent observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import _component, _module_device_dtype, torch_dtype


class Flux2VaeDecoder:
    """Decode canonical TailSteer FLUX.2 latents without loading the DiT or LLM."""

    def __init__(self, *, vae: Any, image_processor: Any, device: Any, dtype: Any):
        self.vae = vae
        self.image_processor = image_processor
        self.device = device
        self.dtype = dtype

    @classmethod
    def load(cls, config: dict[str, Any], *, device: Any) -> Flux2VaeDecoder:
        from diffusers import AutoencoderKLFlux2
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

        model_root = Path(config["path"]).expanduser().resolve()
        vae_source = Path(config.get("vae_path", model_root)).expanduser().resolve()
        vae_root = _component(vae_source, "vae", "config.json")
        if vae_root is None:
            support = config.get("support_path")
            if support is not None:
                vae_root = _component(
                    Path(support).expanduser().resolve(), "vae", "config.json"
                )
        if vae_root is None:
            raise FileNotFoundError(f"missing FLUX.2 VAE under {vae_source}")
        dtype = torch_dtype(str(config.get("dtype", "bfloat16")))
        vae = AutoencoderKLFlux2.from_pretrained(
            str(vae_root), dtype=dtype, local_files_only=True
        ).to(device)
        vae.requires_grad_(False)
        vae.eval()
        scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        processor = Flux2ImageProcessor(vae_scale_factor=scale_factor * 2)
        return cls(vae=vae, image_processor=processor, device=device, dtype=dtype)

    @staticmethod
    def _unpatchify_latents(latents: Any) -> Any:
        batch, channels, height, width = latents.shape
        latents = latents.reshape(batch, channels // 4, 2, 2, height, width).permute(
            0, 1, 4, 2, 5, 3
        )
        return latents.reshape(batch, channels // 4, height * 2, width * 2)

    def decode_latent(self, latent: Any, *, height: int, width: int) -> Any:
        import torch

        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        if latent.ndim != 4 or latent.shape[0] != 1:
            raise ValueError(
                "FLUX.2 visualization expects one [C,H,W] or [1,C,H,W] latent; "
                f"received {tuple(latent.shape)}"
            )
        device, dtype = _module_device_dtype(self.vae, self.device, self.dtype)
        with torch.inference_mode():
            latent = latent.to(device=device, dtype=dtype)
            mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latent)
            std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1)
                + self.vae.config.batch_norm_eps
            ).to(latent)
            unpacked = self._unpatchify_latents(latent * std + mean)
            decoded = self.vae.decode(unpacked, return_dict=False)[0]
            image = self.image_processor.postprocess(decoded, output_type="pil")[0]
        if image.size != (width, height):
            raise ValueError(
                f"decoded image size {image.size} != expected {(width, height)}"
            )
        return image

    def decode_state(self, state: Any, *, width: int, height: int) -> Any:
        """Decode one state using the common cache-backfill decoder contract."""

        return self.decode_latent(state, height=height, width=width)

    def unload(self) -> None:
        """Release the VAE before the caller proceeds to another workload."""

        import gc

        import torch

        del self.vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
