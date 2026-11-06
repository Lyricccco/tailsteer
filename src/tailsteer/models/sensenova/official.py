"""Thin loader for the locked official SenseNova U1.5 source tree."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .identity import SENSENOVA_MODEL_VARIANTS

SENSENOVA_SOURCE_REVISION = "f71dfb098226d01edc0e4c67b3917a2af71a30ef"
SENSENOVA_VARIANTS = tuple(SENSENOVA_MODEL_VARIANTS.values())


@dataclass
class SenseNovaRuntime:
    model: Any
    tokenizer: Any
    source_root: Path
    model_root: Path
    variant: str
    device: str
    _trace_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def unload(self) -> None:
        import torch

        if hasattr(self.model, "to"):
            self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class SenseNovaTrace:
    image: Any
    source_state: Any
    forward_states: dict[int, Any]
    final_state: Any
    timesteps: tuple[float, ...]


def _source_root(model_code_repo: str | Path) -> Path:
    root = Path(model_code_repo).expanduser().resolve()
    package_root = root / "src"
    required = (
        "sensenova_u1/utils/checkpoint_loading.py",
        "sensenova_u1/models/neo_unify/modeling_neo_chat.py",
    )
    if missing := [name for name in required if not (package_root / name).is_file()]:
        raise FileNotFoundError(f"incomplete SenseNova source {root}: {missing}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    return root


def _indexed_weight_files(index_path: Path) -> tuple[str, ...]:
    value = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid model weight index: {index_path}")
    return tuple(sorted({str(name) for name in weight_map.values()}))


def validate_sensenova_snapshot(model_root: str | Path, *, variant: str) -> Path:
    if variant not in SENSENOVA_VARIANTS:
        raise ValueError(f"unsupported SenseNova variant {variant!r}")
    root = Path(model_root).expanduser().resolve()
    required = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    missing = [name for name in required if not (root / name).is_file()]
    index = root / "model.safetensors.index.json"
    single = root / "model.safetensors"
    if not index.is_file() and not single.is_file():
        missing.append("model.safetensors or model.safetensors.index.json")
    if missing:
        raise FileNotFoundError(f"incomplete SenseNova snapshot {root}: {missing}")
    if index.is_file():
        shards = _indexed_weight_files(index)
        if absent := [name for name in shards if not (root / name).is_file()]:
            raise FileNotFoundError(
                f"SenseNova snapshot misses {len(absent)} shards; first={absent[0]}"
            )
    return root


def load_sensenova_runtime(
    *,
    model_root: str | Path,
    model_code_repo: str | Path,
    variant: str,
    device: str = "cuda",
    torch_dtype: Any = None,
) -> SenseNovaRuntime:
    import torch

    source_root = _source_root(model_code_repo)
    root = validate_sensenova_snapshot(model_root, variant=variant)
    from sensenova_u1.utils import load_model_and_tokenizer

    dtype = torch.bfloat16 if torch_dtype is None else torch_dtype
    model, tokenizer = load_model_and_tokenizer(str(root), dtype=dtype, device=device)
    model.eval()
    required = (
        "it2i_generate",
        "patchify",
        "unpatchify",
        "_build_it2i_inputs",
        "_it2i_prefix_forward",
        "_t2i_predict_v",
        "_apply_time_schedule",
    )
    if missing := [
        name for name in required if not callable(getattr(model, name, None))
    ]:
        raise RuntimeError(f"SenseNova model lacks official methods: {missing}")
    return SenseNovaRuntime(
        model=model,
        tokenizer=tokenizer,
        source_root=source_root,
        model_root=root,
        variant=variant,
        device=str(device),
    )


def prepare_sensenova_image(image: Any, *, width: int, height: int) -> Any:
    from PIL import Image

    if width < 1 or height < 1:
        raise ValueError("SenseNova image size must be positive")
    source = image.convert("RGB")
    if source.size == (width, height):
        return source.copy()
    return source.resize((width, height), Image.Resampling.LANCZOS)


def prepare_sensenova_reference_inputs(
    runtime: SenseNovaRuntime, image: Any
) -> tuple[Any, Any]:
    """Run the exact source-image preprocessing used by official ``it2i_generate``."""

    import torch
    from sensenova_u1.models.neo_unify.utils import load_image_native

    model = runtime.model
    pixel_values, grid_hw = load_image_native(
        image,
        int(model.patch_size),
        float(model.downsample_ratio),
        min_pixels=512 * 512,
        max_pixels=min(2048 * 2048, 4096 * 4096),
        upscale=False,
    )
    return (
        pixel_values.to(device=model.device, dtype=torch.bfloat16),
        grid_hw.to(device=model.device),
    )


def _state_patch(model: Any) -> int:
    merge_size = int(1 / float(model.downsample_ratio))
    if merge_size < 1:
        raise ValueError("invalid SenseNova downsample_ratio")
    return int(model.patch_size) * merge_size


def _normalized_pixels(
    image: Any, *, width: int, height: int, device: Any, dtype: Any
) -> Any:
    import numpy as np
    import torch

    prepared = prepare_sensenova_image(image, width=width, height=height)
    array = np.asarray(prepared, dtype=np.float32) / 127.5 - 1.0
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )


def pack_sensenova_image(
    runtime: SenseNovaRuntime, image: Any, *, width: int, height: int
) -> Any:
    parameter = next(runtime.model.parameters())
    pixels = _normalized_pixels(
        image,
        width=width,
        height=height,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    return runtime.model.patchify(pixels, _state_patch(runtime.model))


def decode_sensenova_state(state: Any, *, width: int, height: int) -> Any:
    """Decode one packed pixel state without loading SenseNova model weights."""

    import math

    import numpy as np
    import torch
    from PIL import Image

    value = state.detach().float().cpu()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            "SenseNova packed state must have shape [L,D] or [1,L,D], "
            f"got {tuple(value.shape)}"
        )
    patch_squared = int(value.shape[-1]) // 3
    patch = math.isqrt(patch_squared)
    if patch * patch != patch_squared or int(value.shape[-1]) != patch * patch * 3:
        raise ValueError(f"invalid SenseNova packed channel size: {value.shape[-1]}")
    if height % patch or width % patch:
        raise ValueError(
            f"SenseNova output size {width}x{height} is not aligned to patch {patch}"
        )
    grid_height, grid_width = height // patch, width // patch
    if int(value.shape[1]) != grid_height * grid_width:
        raise ValueError(
            "SenseNova packed token count differs from output grid: "
            f"{value.shape[1]} != {grid_height * grid_width}"
        )
    pixels = value.reshape(1, grid_height, grid_width, patch, patch, 3)
    pixels = torch.einsum("nhwpqc->nchpwq", pixels)
    pixels = pixels.reshape(1, 3, height, width)
    array = (
        ((pixels[0].clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .permute(1, 2, 0)
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(np.asarray(array)).convert("RGB")


def sensenova_timesteps(
    runtime: SenseNovaRuntime, *, width: int, height: int, steps: int, shift: float
) -> Any:
    import torch

    model = runtime.model
    patch = _state_patch(model)
    token_count = (height // patch) * (width // patch)
    values = torch.linspace(0.0, 1.0, steps + 1, device=runtime.device)
    return model._apply_time_schedule(values, token_count, shift)


def run_sensenova_edit(
    runtime: SenseNovaRuntime,
    *,
    source_image: Any,
    prompt: str,
    seed: int,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 50,
    cfg_scale: float = 4.0,
    img_cfg_scale: float = 1.0,
    cfg_norm: str = "none",
    shift: float = 3.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    capture_forward_steps: Sequence[int] = (),
) -> SenseNovaTrace:
    """Run the official sampler while observing exact packed-state boundaries."""

    import numpy as np
    import torch
    from PIL import Image

    if num_inference_steps < 1:
        raise ValueError("SenseNova inference steps must be positive")
    patch = _state_patch(runtime.model)
    if width % patch or height % patch:
        raise ValueError(
            f"SenseNova size {width}x{height} is not aligned to state patch {patch}"
        )
    requested = {int(step) for step in capture_forward_steps}
    if any(step < 1 or step > num_inference_steps for step in requested):
        raise ValueError("SenseNova capture steps are one-based and in-range")

    prepared = prepare_sensenova_image(source_image, width=width, height=height)
    prompt = str(prompt)
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    captured: dict[int, Any] = {}
    seen_timesteps: set[float] = set()
    forward_count = 0
    model = runtime.model
    original_predict = model._t2i_predict_v

    def traced_predict(*args: Any, **kwargs: Any):
        nonlocal forward_count
        timestep = kwargs.get("t", args[4] if len(args) > 4 else None)
        state = kwargs.get("z", args[5] if len(args) > 5 else None)
        if timestep is None or state is None:
            raise RuntimeError("SenseNova predictor did not expose timestep/state")
        timestep_key = round(float(timestep.detach().float().cpu()), 9)
        if timestep_key not in seen_timesteps:
            seen_timesteps.add(timestep_key)
            forward_count += 1
            if forward_count in requested:
                captured[forward_count] = state.detach().cpu().contiguous()
        return original_predict(*args, **kwargs)

    with runtime._trace_lock:
        model._t2i_predict_v = traced_predict
        try:
            output = model.it2i_generate(
                runtime.tokenizer,
                prompt,
                [prepared],
                image_size=(int(width), int(height)),
                cfg_scale=float(cfg_scale),
                img_cfg_scale=float(img_cfg_scale),
                cfg_norm=str(cfg_norm),
                enable_timestep_shift=True,
                timestep_shift=float(shift),
                cfg_interval=tuple(float(value) for value in cfg_interval),
                num_steps=int(num_inference_steps),
                batch_size=1,
                think_mode=False,
                seed=int(seed),
            )
        finally:
            model._t2i_predict_v = original_predict

    if forward_count != num_inference_steps:
        raise RuntimeError(
            "SenseNova official rollout exposed "
            f"{forward_count} unique denoise steps, expected {num_inference_steps}"
        )
    if missing := requested - set(captured):
        raise RuntimeError(f"SenseNova rollout missed forwards {sorted(missing)}")
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        raise RuntimeError("SenseNova it2i_generate returned an invalid pixel tensor")

    final_state = model.patchify(output, patch).detach().cpu().contiguous()
    source_state = (
        pack_sensenova_image(runtime, prepared, width=width, height=height)
        .detach()
        .cpu()
        .contiguous()
    )
    if final_state.shape != source_state.shape:
        raise RuntimeError(
            "SenseNova source/final state shapes differ: "
            f"{tuple(source_state.shape)} vs {tuple(final_state.shape)}"
        )
    array = ((output[0].float().clamp(-1, 1) + 1.0) * 127.5).round()
    array = array.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(np.asarray(array)).convert("RGB")
    timesteps = sensenova_timesteps(
        runtime,
        width=width,
        height=height,
        steps=num_inference_steps,
        shift=shift,
    )
    return SenseNovaTrace(
        image=image,
        source_state=source_state,
        forward_states=captured,
        final_state=final_state,
        timesteps=tuple(float(value) for value in timesteps.detach().cpu()),
    )
