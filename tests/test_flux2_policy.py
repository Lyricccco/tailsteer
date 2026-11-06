from __future__ import annotations

import json

import pytest
import torch

from tailsteer.models.flux2.model import Flux2Model
from tailsteer.models.flux2.output_adapter import (
    Flux2OutputAdapterConfig,
    attach_flux2_output_adapter,
)
from tailsteer.models.flux2.policy import (
    canonical_flux2_policy,
    configure_flux2_structured_trainable,
)


class TinyFluxTransformer(torch.nn.Module):
    def __init__(self, *, single_blocks: int = 4, double_blocks: int = 2) -> None:
        super().__init__()
        self.double_stream_modulation_img = torch.nn.Linear(2, 2)
        self.double_stream_modulation_txt = torch.nn.Linear(2, 2)
        self.single_stream_modulation = torch.nn.Linear(2, 2)
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(double_blocks)]
        )
        self.single_transformer_blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(single_blocks)]
        )
        self.norm_out = torch.nn.LayerNorm(2)
        self.proj_out = torch.nn.Linear(2, 2)


def _trainable_names(root: torch.nn.Module) -> set[str]:
    return {
        name for name, parameter in root.named_parameters() if parameter.requires_grad
    }


@pytest.mark.parametrize(
    ("policy", "expected_prefixes"),
    [
        ("output_head", ("proj_out",)),
        ("output_head_norm", ("norm_out", "proj_out")),
        (
            "tail_single",
            (
                "norm_out",
                "proj_out",
                "single_stream_modulation",
                "single_transformer_blocks.2",
                "single_transformer_blocks.3",
            ),
        ),
        (
            "single_stream",
            (
                "norm_out",
                "proj_out",
                "single_stream_modulation",
                "single_transformer_blocks",
            ),
        ),
        (
            "dit_full",
            (
                "double_stream_modulation_img",
                "double_stream_modulation_txt",
                "single_stream_modulation",
                "transformer_blocks",
                "single_transformer_blocks",
                "norm_out",
                "proj_out",
            ),
        ),
    ],
)
def test_structured_policy_selects_only_declared_modules(
    policy: str, expected_prefixes: tuple[str, ...]
) -> None:
    root = TinyFluxTransformer()
    selection = configure_flux2_structured_trainable(root, {"policy": policy})

    assert selection.policy == policy
    assert all(
        any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in expected_prefixes
        )
        for name in _trainable_names(root)
    )
    assert all(
        any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in expected_prefixes
        )
        for name in selection.parameter_names
    )
    assert len(_trainable_names(root)) == len(selection.parameter_names)


def test_structured_policy_alias_and_tail_size() -> None:
    assert canonical_flux2_policy("H4") == "dit_full"
    root = TinyFluxTransformer(single_blocks=5)
    selection = configure_flux2_structured_trainable(
        root, {"policy": "h2", "tail_blocks": 1}
    )

    assert selection.policy == "tail_single"
    assert selection.tail_blocks == 1
    assert "single_transformer_blocks[-1:]" in selection.groups
    assert any(
        name.startswith("single_transformer_blocks.4.")
        for name in selection.parameter_names
    )
    assert not any(
        name.startswith("single_transformer_blocks.3.")
        for name in selection.parameter_names
    )


def test_fp32_master_keeps_structured_policy_selective() -> None:
    root = TinyFluxTransformer().to(dtype=torch.bfloat16)
    model = Flux2Model.__new__(Flux2Model)
    model.refined_transformer = root
    model.configure_trainable({"policy": "output_head_norm", "master_dtype": "float32"})
    model.prepare_optimizer_parameters()

    assert all(parameter.dtype == torch.float32 for parameter in root.parameters())
    assert all(
        name.startswith(("norm_out.", "proj_out."))
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in root.named_parameters()
        if not name.startswith(("norm_out.", "proj_out."))
    )


def test_structured_policy_requires_official_module_names() -> None:
    root = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="proj_out"):
        configure_flux2_structured_trainable(root, {"policy": "output_head"})


def test_residual_output_adapter_is_zero_initialized_and_keeps_projection_keys() -> (
    None
):
    root = TinyFluxTransformer()
    original_weight = root.proj_out.weight.detach().clone()
    original_input = torch.randn(2, 3, 2)
    expected = torch.nn.functional.linear(
        original_input, root.proj_out.weight, root.proj_out.bias
    )

    resolved = attach_flux2_output_adapter(
        root,
        {"enabled": True, "type": "residual_mlp", "bottleneck": 128},
    )

    assert resolved == Flux2OutputAdapterConfig(
        enabled=True, type="residual_mlp", bottleneck=128
    )
    torch.testing.assert_close(root.proj_out.weight, original_weight)
    torch.testing.assert_close(root.proj_out(original_input), expected)
    names = set(root.state_dict())
    assert "proj_out.weight" in names
    assert "proj_out.tailsteer_adapter.0.weight" in names
    assert "proj_out.tailsteer_adapter.2.weight" in names
    assert torch.count_nonzero(root.proj_out.tailsteer_adapter[2].weight) == 0


def test_h1_policy_includes_residual_output_adapter() -> None:
    root = TinyFluxTransformer().to(dtype=torch.bfloat16)
    model = Flux2Model.__new__(Flux2Model)
    model.refined_transformer = root
    model.configure_trainable(
        {
            "policy": "output_head_norm",
            "master_dtype": "float32",
            "output_adapter": {
                "enabled": True,
                "type": "residual_mlp",
                "bottleneck": 128,
            },
        }
    )

    assert "output_residual_adapter" in model.trainable_groups
    assert model.trainable_config["output_adapter"]["bottleneck"] == 128
    assert any(
        name.startswith("proj_out.tailsteer_adapter.")
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    )
    assert all(parameter.requires_grad for parameter in root.norm_out.parameters())


def test_output_adapter_rejects_lora_policy() -> None:
    root = TinyFluxTransformer()
    model = Flux2Model.__new__(Flux2Model)
    model.refined_transformer = root
    with pytest.raises(ValueError, match="not supported with the lora policy"):
        model.configure_trainable(
            {
                "policy": "lora",
                "output_adapter": {"enabled": True},
            }
        )


def test_structured_flux_export_records_policy_groups(tmp_path) -> None:
    root = TinyFluxTransformer()
    model = Flux2Model.__new__(Flux2Model)
    model.refined_transformer = root
    model.configure_trainable({"policy": "dit_full", "master_dtype": "model"})

    state, weights, manifest = model.prepare_inference_export(
        root.state_dict(), step=1000
    )

    assert weights == "transformer.safetensors"
    assert set(state) == set(root.state_dict())
    assert manifest["trainable_policy"] == "dit_full"
    assert manifest["trainable_groups"]
    assert manifest["trainable"]["policy"] == "dit_full"

    # The metadata is the same JSON contract written by the checkpoint
    # exporter; keeping this check here catches non-serializable module data.
    json.dumps(manifest)

    from safetensors.torch import save_file

    export = tmp_path / "export"
    export.mkdir()
    save_file(state, export / weights)
    (export / "manifest.json").write_text(json.dumps(manifest))
    (export / "COMPLETE").write_text("complete\n")

    restored = Flux2Model.__new__(Flux2Model)
    restored.refined_transformer = TinyFluxTransformer()
    restored.load_export(export)
    assert restored.trainable_policy == "dit_full"
    assert restored.trainable_groups == model.trainable_groups
    for name, value in restored.refined_transformer.state_dict().items():
        torch.testing.assert_close(value, state[name])


def test_residual_output_adapter_export_reconstructs_before_loading(tmp_path) -> None:
    root = TinyFluxTransformer()
    adapter_config = {
        "enabled": True,
        "type": "residual_mlp",
        "bottleneck": 128,
        "activation": "silu",
        "zero_init_output": True,
    }
    attach_flux2_output_adapter(root, adapter_config)
    model = Flux2Model.__new__(Flux2Model)
    model.refined_transformer = root
    model.trainable_policy = "output_head_norm"
    model.trainable_groups = (
        "terminal_norm",
        "output_head",
        "output_residual_adapter",
    )
    model.trainable_config = {
        "policy": "output_head_norm",
        "master_dtype": "float32",
        "output_adapter": adapter_config,
        "trainable_groups": list(model.trainable_groups),
    }
    state = root.state_dict()
    export = tmp_path / "export"
    export.mkdir()
    from safetensors.torch import save_file

    save_file(state, export / "transformer.safetensors")
    (export / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "flux2_klein_9b",
                "step": 1,
                "trainable_policy": "output_head_norm",
                "trainable": model.trainable_config,
                "trainable_groups": list(model.trainable_groups),
                "weights": "transformer.safetensors",
            }
        )
    )
    (export / "COMPLETE").write_text("complete\n")

    restored = Flux2Model.__new__(Flux2Model)
    restored.refined_transformer = TinyFluxTransformer()
    restored.load_export(export)

    assert hasattr(restored.refined_transformer.proj_out, "tailsteer_adapter")
    for name, value in restored.refined_transformer.state_dict().items():
        torch.testing.assert_close(value, state[name])
