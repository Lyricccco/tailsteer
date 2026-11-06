from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from tailsteer.models.sensenova.model import SenseNovaModel
from tailsteer.models.sensenova.training import (
    configure_sensenova_trainable,
    prepare_sensenova_reference_inputs,
)


class _TrainableSenseNova(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = torch.nn.Linear(2, 2, bias=False)
        self.language_model = torch.nn.Module()
        self.language_model.model = torch.nn.Module()
        self.language_model.model.q_proj_mot_gen = torch.nn.Linear(2, 2, bias=False)
        self.fm_modules = torch.nn.ModuleDict(
            {"fm_head": torch.nn.Linear(2, 2, bias=False)}
        )


def test_sensenova_default_trainable_uses_fp32_master_parameters() -> None:
    raw = _TrainableSenseNova().to(dtype=torch.bfloat16)
    selected = configure_sensenova_trainable(raw, {"policy": "generation_denoiser"})
    assert selected
    assert raw._tailsteer_master_dtype == "float32"

    root = torch.nn.Module()
    root.model = raw
    backend = SenseNovaModel.__new__(SenseNovaModel)
    backend._root = root
    backend.trainable_config = {
        "policy": "generation_denoiser",
        "master_dtype": "float32",
    }
    backend._fsdp_master_parameters_prepared = False
    backend.prepare_optimizer_parameters()

    assert all(
        parameter.dtype == torch.float32
        for parameter in raw.parameters()
        if parameter.requires_grad
    )


def test_sensenova_conditioning_uses_original_image_and_official_preprocess(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (23, 31), color=(12, 34, 56)).save(source)
    expected_pixels = torch.randn(4, 8)
    expected_grid = torch.tensor([[2, 2]])
    observed = {}

    def official_prepare(runtime, image):
        observed["runtime"] = runtime
        observed["size"] = image.size
        observed["pixel"] = image.getpixel((0, 0))
        return expected_pixels, expected_grid

    monkeypatch.setattr(
        "tailsteer.models.sensenova.official.prepare_sensenova_reference_inputs",
        official_prepare,
    )
    runtime = SimpleNamespace(model=SimpleNamespace())
    actual = prepare_sensenova_reference_inputs(
        runtime,
        metadata_item={
            "source_path": str(source),
            "width": 32,
            "height": 32,
        },
        device=torch.device("cpu"),
    )

    assert observed == {
        "runtime": runtime,
        "size": (32, 32),
        "pixel": (12, 34, 56),
    }
    assert actual.pixel_values is expected_pixels
    assert actual.grid_hw is expected_grid
