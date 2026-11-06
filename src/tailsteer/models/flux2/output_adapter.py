"""Optional residual output-capacity adapters for the official FLUX.2 head."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Flux2OutputAdapterConfig:
    """Validated configuration for the opt-in FLUX.2 output adapter."""

    enabled: bool = False
    type: str = "residual_mlp"
    bottleneck: int = 128
    activation: str = "silu"
    zero_init_output: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> Flux2OutputAdapterConfig:
        raw = {} if value is None else dict(value)
        enabled = bool(raw.get("enabled", False))
        adapter_type = str(raw.get("type", "residual_mlp")).strip().lower()
        if adapter_type != "residual_mlp":
            raise ValueError("FLUX.2 output_adapter.type must be 'residual_mlp'")
        bottleneck = raw.get("bottleneck", 128)
        if isinstance(bottleneck, bool) or not isinstance(bottleneck, int):
            raise TypeError("FLUX.2 output_adapter.bottleneck must be an integer")
        if bottleneck < 1:
            raise ValueError("FLUX.2 output_adapter.bottleneck must be positive")
        activation = str(raw.get("activation", "silu")).strip().lower()
        if activation != "silu":
            raise ValueError("FLUX.2 output_adapter.activation must be 'silu'")
        zero_init = raw.get("zero_init_output", True)
        if not isinstance(zero_init, bool):
            raise TypeError("FLUX.2 output_adapter.zero_init_output must be a boolean")
        if enabled and not zero_init:
            raise ValueError(
                "FLUX.2 residual output adapters require zero_init_output=true"
            )
        return cls(
            enabled=enabled,
            type=adapter_type,
            bottleneck=bottleneck,
            activation=activation,
            zero_init_output=zero_init,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "type": self.type,
            "bottleneck": self.bottleneck,
            "activation": self.activation,
            "zero_init_output": self.zero_init_output,
        }


def _linear_dimensions(module: Any) -> tuple[int, int, Any, Any]:
    import torch.nn as nn

    if not isinstance(module, nn.Linear):
        raise TypeError(
            "FLUX.2 residual output adapter requires proj_out to be nn.Linear; "
            f"got {type(module).__name__}"
        )
    return module.in_features, module.out_features, module.weight, module.bias


def _build_residual_mlp(
    input_dim: int, output_dim: int, bottleneck: int, *, device: Any, dtype: Any
) -> Any:
    import torch.nn as nn

    adapter = nn.Sequential(
        nn.Linear(input_dim, bottleneck, bias=True),
        nn.SiLU(),
        nn.Linear(bottleneck, output_dim, bias=False),
    ).to(device=device, dtype=dtype)
    # The adapter must be an exact no-op when it is attached to a pretrained
    # transformer.  The first layer can be random, but the output projection
    # is zero so no residual reaches the original velocity path at step zero.
    nn.init.zeros_(adapter[-1].weight)
    return adapter


def build_residual_output_projection(base: Any, config: Mapping[str, Any]) -> Any:
    """Wrap a FLUX.2 linear ``proj_out`` with a zero-init residual MLP.

    The wrapper keeps ``weight`` and ``bias`` directly under ``proj_out`` so
    old transformer checkpoints retain their original key names.  New adapter
    weights are stored below ``proj_out.tailsteer_adapter``.  The official
    transformer still owns the complete forward graph; only the final linear
    projection is extended.
    """

    values = Flux2OutputAdapterConfig.from_mapping(config)
    if not values.enabled:
        return base
    if hasattr(base, "tailsteer_adapter"):
        return base
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    input_dim, output_dim, weight, bias = _linear_dimensions(base)

    class _Projection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Register the original parameters at the wrapper's top level to
            # preserve proj_out.weight/proj_out.bias checkpoint keys.
            self.weight = weight
            self.register_parameter("bias", bias)
            self.in_features = input_dim
            self.out_features = output_dim
            self.tailsteer_adapter = _build_residual_mlp(
                input_dim,
                output_dim,
                values.bottleneck,
                device=weight.device,
                dtype=weight.dtype,
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            base_output = F.linear(hidden_states, self.weight, self.bias)
            return base_output + self.tailsteer_adapter(hidden_states)

    return _Projection()


def attach_flux2_output_adapter(
    root: Any, config: Mapping[str, Any] | None
) -> Flux2OutputAdapterConfig:
    """Attach the optional adapter to an unwrapped FLUX.2 transformer root."""

    values = Flux2OutputAdapterConfig.from_mapping(config)
    if not values.enabled:
        return values
    if values.type != "residual_mlp":
        raise ValueError(f"unsupported FLUX.2 output adapter {values.type!r}")
    projection = getattr(root, "proj_out", None)
    if projection is None:
        raise ValueError("FLUX.2 transformer has no required module 'proj_out'")
    if hasattr(projection, "tailsteer_adapter"):
        return values
    root.proj_out = build_residual_output_projection(projection, values.to_dict())
    return values


def flux2_output_adapter_config(
    config: Mapping[str, Any] | None,
) -> Flux2OutputAdapterConfig:
    """Resolve the optional adapter configuration without attaching it."""

    return Flux2OutputAdapterConfig.from_mapping(config)
