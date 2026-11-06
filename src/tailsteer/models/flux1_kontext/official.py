"""Thin local loader and read-only capture around BFL's original denoise loop."""

from __future__ import annotations

import io
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KONTEXT_SOURCE_REVISION = "802fb4713906133fcbd0d8dc5351620ca4773036"


def import_source(code_path):
    root = Path(code_path).resolve() / "src"
    if not (root / "flux/sampling.py").is_file():
        raise FileNotFoundError(root)
    existing = sys.modules.get("flux")
    if existing is not None and root not in Path(existing.__file__).resolve().parents:
        raise RuntimeError("another flux package is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_component(cls, params, path, device, *, dtype=None):
    import torch
    from safetensors.torch import load_file

    with torch.device("meta"):
        module = cls(params)
        if dtype is not None:
            module = module.to(dtype)
    # As in BFL loaders, assign preserves checkpoint precision (AE is FP32).
    module.load_state_dict(
        load_file(str(path), device=str(device)), strict=True, assign=True
    )
    return module.eval().requires_grad_(False)


def _load_embedder(root, *, is_clip, device):
    import torch
    from flux.modules.conditioner import HFEmbedder
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

    class LocalEmbedder(HFEmbedder):
        """Only resolve local component paths; inherit BFL's forward unchanged."""

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.is_clip = is_clip
            self.max_length = 77 if is_clip else 512
            self.output_key = "pooler_output" if is_clip else "last_hidden_state"
            suffix = "" if is_clip else "_2"
            tokenizer = CLIPTokenizer if is_clip else T5Tokenizer
            encoder = CLIPTextModel if is_clip else T5EncoderModel
            self.tokenizer = tokenizer.from_pretrained(
                str(root / f"tokenizer{suffix}"),
                max_length=self.max_length,
                local_files_only=True,
            )
            self.hf_module = (
                encoder.from_pretrained(
                    str(root / f"text_encoder{suffix}"),
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
                .eval()
                .requires_grad_(False)
            )

    return LocalEmbedder().to(device)


def unpack_state(tokens, *, height, width):
    import torch
    from flux.sampling import unpack

    if tokens.ndim != 3 or tokens.shape != (1, height * width // 256, 64):
        raise ValueError(f"unexpected Kontext target tokens: {tokens.shape}")
    state = unpack(tokens, height, width)[0].detach().cpu().clone()
    if not torch.isfinite(state).all():
        raise RuntimeError("non-finite Kontext state")
    return state


@dataclass
class KontextTrace:
    image: Any = None
    source_state: Any = None
    forward_states: dict[int, Any] = field(default_factory=dict)
    final_state: Any = None
    schedule_points: dict[int, tuple[float, float]] = field(default_factory=dict)
    final_schedule_point: tuple[float, float] = (0.0, 0.0)


@contextmanager
def capture_states(model, *, points, sigmas, height, width):
    """Observe target tokens before each forward; never replace the Euler update."""
    if not points or tuple(points) != tuple(sorted(set(points))):
        raise ValueError("capture points must be non-empty, unique and increasing")
    if min(points) < 0 or max(points) >= len(sigmas) - 1:
        raise ValueError("capture point outside the forward range")
    trace = KontextTrace()
    calls = 0
    target_tokens = height * width // 256

    def observe(module, args, kwargs):
        nonlocal calls
        if calls >= len(sigmas) - 1:
            raise RuntimeError("too many Kontext forwards")
        if calls in points:
            trace.forward_states[calls] = unpack_state(
                kwargs["img"][:, :target_tokens], height=height, width=width
            )
            # Native model input t is BF16 sigma, not Diffusers' t*1000.
            trace.schedule_points[calls] = (
                float(sigmas[calls]),
                float(kwargs["timesteps"][0]),
            )
        calls += 1

    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        yield trace
        if calls != len(sigmas) - 1:
            raise RuntimeError(f"incomplete Kontext trajectory: {calls} forwards")
    finally:
        handle.remove()


def decode_image(ae, state, *, device):
    import torch
    from PIL import Image

    # Same AE call as cli_kontext, before its watermark/JPEG export layer.
    with (
        torch.no_grad(),
        torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16),
    ):
        pixels = ae.decode(state[None].to(device=device, dtype=torch.float32))
    pixels = pixels[0].float().clamp(-1, 1)
    return Image.fromarray(
        (127.5 * (pixels + 1)).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )


class KontextRuntime:
    @classmethod
    def load(cls, model, *, device):
        import torch

        import_source(model["code_path"])
        from flux.model import Flux
        from flux.modules.autoencoder import AutoEncoder
        from flux.util import configs

        if model.get("dtype", "bfloat16") != "bfloat16":
            raise ValueError(
                "Kontext cache preserves the native BF16 DiT configuration"
            )
        root = Path(model["path"]).resolve()
        result = cls()
        result.device = torch.device(device)
        result._lock = threading.RLock()
        spec = configs["flux-dev-kontext"]
        result.model = _load_component(
            Flux,
            spec.params,
            root / "flux1-kontext-dev.safetensors",
            device,
            dtype=torch.bfloat16,
        )
        result.ae = _load_component(
            AutoEncoder, spec.ae_params, root / "ae.safetensors", device
        )
        result.t5 = _load_embedder(root, is_clip=False, device=device)
        result.clip = _load_embedder(root, is_clip=True, device=device)
        return result

    def edit(
        self,
        *,
        source_image,
        prompt,
        seed,
        num_steps,
        guidance_scale,
        capture_forward_steps,
    ):
        import torch
        from flux.sampling import denoise, get_schedule, prepare_kontext

        if source_image.size != (1024, 1024):
            raise ValueError("Kontext cache expects the shared 1024x1024 preprocessing")
        with self._lock, torch.inference_mode(), io.BytesIO() as source_buffer:
            source_image.save(source_buffer, format="PNG")
            source_buffer.seek(0)
            inputs, height, width = prepare_kontext(
                self.t5,
                self.clip,
                prompt,
                self.ae,
                source_buffer,
                seed=seed,
                device=self.device,
                target_width=1024,
                target_height=1024,
            )
            inputs.pop("img_cond_orig")
            sigmas = get_schedule(num_steps, inputs["img"].shape[1], shift=True)
            with capture_states(
                self.model,
                points=capture_forward_steps,
                sigmas=sigmas,
                height=height,
                width=width,
            ) as trace:
                trace.source_state = unpack_state(
                    inputs["img_cond_seq"], height=height, width=width
                )
                final = denoise(
                    self.model, **inputs, timesteps=sigmas, guidance=guidance_scale
                )
            trace.final_state = unpack_state(final, height=height, width=width)
            trace.final_schedule_point = (float(sigmas[-1]), 0.0)
            trace.image = decode_image(self.ae, trace.final_state, device=self.device)
        return trace

    def unload(self):
        self.model = self.ae = self.t5 = self.clip = None


class KontextVaeDecoder:
    @classmethod
    def load(cls, model, *, device):
        import_source(model["code_path"])
        from flux.modules.autoencoder import AutoEncoder
        from flux.util import configs

        result = cls()
        result.device = device
        result.ae = _load_component(
            AutoEncoder,
            configs["flux-dev-kontext"].ae_params,
            Path(model["path"]) / "ae.safetensors",
            device,
        )
        return result

    def decode_state(self, state, *, width, height):
        if tuple(state.shape) != (16, height // 8, width // 8):
            raise ValueError(f"unexpected Kontext latent shape: {state.shape}")
        return decode_image(self.ae, state, device=self.device)

    def unload(self):
        self.ae = None
