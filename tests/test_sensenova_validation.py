from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from tailsteer.models.sensenova.model import SenseNovaModel
from tailsteer.training.visual_validation import (
    image_edit_metrics,
    select_fixed_local_edit_indices,
)


class _Records:
    def __init__(self) -> None:
        self.records = tuple(
            {
                "sample_id": f"sample-{index}",
                "source_identity": f"source-{index}",
                "task": "local_edit" if index != 2 else "noop",
            }
            for index in range(5)
        )

    def __len__(self) -> int:
        return len(self.records)


def test_fixed_visual_selection_only_uses_local_edits() -> None:
    dataset = _Records()

    first = select_fixed_local_edit_indices(dataset, count=3, seed=7)
    second = select_fixed_local_edit_indices(dataset, count=3, seed=7)

    assert first == second
    assert len(first) == 3
    assert all(dataset.records[index]["task"] == "local_edit" for index in first)


def test_image_edit_metrics_separate_edit_and_keep_regions() -> None:
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    generated = source.copy()
    generated[:, :4] = 255
    keep_mask = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.float32)

    metrics = image_edit_metrics(
        Image.fromarray(source), Image.fromarray(generated), keep_mask
    )

    assert metrics["edit_region_change"]["rgb_mae"] == 1.0
    assert metrics["edit_region_change"]["changed_pixel_ratio"] == 1.0
    assert metrics["non_edit_region_preservation"]["rgb_mae"] == 0.0
    assert metrics["non_edit_region_preservation"]["unchanged_pixel_ratio"] == 1.0
    assert not metrics["full_inference_image_quality"]["semantic_reference_available"]


class _DeltaRoot(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.q_proj = torch.nn.Linear(2, 2, bias=False)
        self.model.language_model.q_proj_mot_gen = torch.nn.Linear(2, 2, bias=False)
        self.model.fm_modules = torch.nn.ModuleDict(
            {"fm_head": torch.nn.Linear(2, 2, bias=False)}
        )


def test_sensenova_delta_loads_every_generation_tensor(tmp_path) -> None:
    model = SenseNovaModel.__new__(SenseNovaModel)
    model.runtime = SimpleNamespace(variant="sensenova-u1.5-8b-mot-preview")
    model.model_name = "sensenova_u1_5_8b_mot_preview"
    model._root = _DeltaRoot()
    expected = {
        name: torch.full_like(tensor, 3.0)
        for name, tensor in model._root.state_dict().items()
        if model._is_generation_state(name)
    }
    save_file(expected, tmp_path / "model.safetensors")
    manifest = {
        "model": "sensenova_u1_5_8b_mot_preview",
        "variant": "sensenova-u1.5-8b-mot-preview",
        "weights": "model.safetensors",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "COMPLETE").write_text("complete\n", encoding="utf-8")

    ordinary_before = model._root.model.language_model.q_proj.weight.detach().clone()
    model.load_export(tmp_path)

    for name, tensor in model._root.state_dict().items():
        if model._is_generation_state(name):
            torch.testing.assert_close(tensor, expected[name])
    torch.testing.assert_close(
        model._root.model.language_model.q_proj.weight, ordinary_before
    )
