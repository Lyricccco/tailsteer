"""FLUX.2 trainable-module policies.

The policy names in this module describe *which* part of the official
transformer is trainable.  They do not alter the forward graph or any
TailSteer objective.  ``full`` and ``lora`` remain owned by the historical
FLUX.2 adapter; the structured policies below are the H0--H4 ablations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

FLUX2_STRUCTURED_POLICIES = (
    "output_head",
    "output_head_norm",
    "tail_single",
    "single_stream",
    "dit_full",
)

_ALIASES = {
    "h0": "output_head",
    "h1": "output_head_norm",
    "h2": "tail_single",
    "h3": "single_stream",
    "h4": "dit_full",
}


@dataclass(frozen=True)
class Flux2PolicySelection:
    """Resolved trainable parameter groups for one FLUX.2 policy."""

    policy: str
    groups: tuple[str, ...]
    tail_blocks: int | None
    parameter_names: tuple[str, ...]

    def config(self) -> dict[str, Any]:
        """Return the serializable policy information stored in manifests."""

        value: dict[str, Any] = {
            "policy": self.policy,
            "trainable_groups": list(self.groups),
        }
        if self.tail_blocks is not None:
            value["tail_blocks"] = self.tail_blocks
        return value


def canonical_flux2_policy(value: str) -> str:
    """Normalize a descriptive policy name or its H0--H4 alias."""

    policy = str(value).strip().lower()
    return _ALIASES.get(policy, policy)


def is_flux2_structured_policy(value: str) -> bool:
    return canonical_flux2_policy(value) in FLUX2_STRUCTURED_POLICIES


def _has_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def _require_module(root: Any, attribute: str) -> Any:
    value = getattr(root, attribute, None)
    if value is None:
        raise ValueError(f"FLUX.2 transformer has no required module {attribute!r}")
    return value


def configure_flux2_structured_trainable(
    root: Any, config: Mapping[str, Any]
) -> Flux2PolicySelection:
    """Apply an H0--H4 policy to an unwrapped FLUX.2 transformer.

    The resolver intentionally uses official module attributes and named
    parameter prefixes.  This keeps the trainer independent of Diffusers
    internals while making a missing architecture component fail loudly.
    """

    requested = str(config.get("policy", ""))
    policy = canonical_flux2_policy(requested)
    if policy not in FLUX2_STRUCTURED_POLICIES:
        raise ValueError(
            "unknown FLUX.2 structured policy "
            f"{requested!r}; available={list(FLUX2_STRUCTURED_POLICIES)}"
        )

    _require_module(root, "proj_out")
    prefixes = ["proj_out"]
    groups = ["output_head"]
    tail_blocks: int | None = None

    if policy != "output_head":
        _require_module(root, "norm_out")
        prefixes.insert(0, "norm_out")
        groups.insert(0, "terminal_norm")

    if policy in {"tail_single", "single_stream", "dit_full"}:
        _require_module(root, "single_stream_modulation")
        _require_module(root, "single_transformer_blocks")
        single_blocks = tuple(root.single_transformer_blocks)
        if not single_blocks:
            raise ValueError("FLUX.2 transformer has no single-stream blocks")
        prefixes.append("single_stream_modulation")
        groups.append("single_stream_modulation")

        if policy == "tail_single":
            tail_blocks = int(config.get("tail_blocks", 2))
            if not 1 <= tail_blocks <= len(single_blocks):
                raise ValueError(
                    "FLUX.2 tail_blocks must be in [1, number of single blocks]; "
                    f"got {tail_blocks}, number={len(single_blocks)}"
                )
            start = len(single_blocks) - tail_blocks
            prefixes.extend(
                f"single_transformer_blocks.{index}"
                for index in range(start, len(single_blocks))
            )
            groups.append(f"single_transformer_blocks[-{tail_blocks}:]")
        else:
            prefixes.append("single_transformer_blocks")
            groups.append("single_transformer_blocks[:]")

    if policy == "dit_full":
        _require_module(root, "double_stream_modulation_img")
        _require_module(root, "double_stream_modulation_txt")
        _require_module(root, "transformer_blocks")
        if not tuple(root.transformer_blocks):
            raise ValueError("FLUX.2 transformer has no double-stream blocks")
        prefixes.extend(
            (
                "double_stream_modulation_img",
                "double_stream_modulation_txt",
                "transformer_blocks",
            )
        )
        groups.extend(
            (
                "double_stream_modulation_img",
                "double_stream_modulation_txt",
                "transformer_blocks[:]",
            )
        )

    # Apply the selection only after all structural validation so an invalid
    # policy cannot leave a partially configured model.
    names = tuple(root.named_parameters())
    selected_names = tuple(
        name
        for name, _parameter in names
        if any(_has_prefix(name, prefix) for prefix in prefixes)
    )
    if not selected_names:
        raise RuntimeError(
            f"FLUX.2 structured policy selected no parameters; prefixes={prefixes}"
        )
    selected_ids = set(selected_names)
    for name, parameter in names:
        parameter.requires_grad_(name in selected_ids)
    return Flux2PolicySelection(
        policy=policy,
        groups=tuple(groups),
        tail_blocks=tail_blocks,
        parameter_names=selected_names,
    )
