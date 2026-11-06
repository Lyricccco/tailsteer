"""Structured trainable policies for the SenseNova generation path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SENSENOVA_STRUCTURED_POLICIES = ("output_head", "output_head_norm")

_ALIASES = {
    "h0": "output_head",
    "h1": "output_head_norm",
}


@dataclass(frozen=True)
class SenseNovaPolicySelection:
    """Resolved parameter groups for a structured SenseNova policy."""

    policy: str
    groups: tuple[str, ...]
    parameter_names: tuple[str, ...]

    def config(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "trainable_groups": list(self.groups),
        }


def canonical_sensenova_policy(value: str) -> str:
    """Normalize descriptive policy names and H0/H1 aliases."""

    policy = str(value).strip().lower()
    return _ALIASES.get(policy, policy)


def is_sensenova_structured_policy(value: str) -> bool:
    return canonical_sensenova_policy(value) in SENSENOVA_STRUCTURED_POLICIES


def _has_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def _require_path(root: Any, path: str) -> Any:
    value = root
    for component in path.split("."):
        value = getattr(value, component, None)
        if value is None:
            raise ValueError(f"SenseNova model has no required module path {path!r}")
    return value


def configure_sensenova_structured_trainable(
    model: Any, config: Mapping[str, Any]
) -> SenseNovaPolicySelection:
    """Apply the H0/H1 output-tail policies without changing the forward."""

    requested = str(config.get("policy", ""))
    policy = canonical_sensenova_policy(requested)
    if policy not in SENSENOVA_STRUCTURED_POLICIES:
        raise ValueError(
            "unknown SenseNova structured policy "
            f"{requested!r}; available={list(SENSENOVA_STRUCTURED_POLICIES)}"
        )

    head_path = "fm_modules.fm_head"
    _require_path(model, head_path)
    prefixes = [head_path]
    groups = ["output_head"]
    if policy == "output_head_norm":
        norm_path = "language_model.model.norm_mot_gen"
        _require_path(model, norm_path)
        prefixes.insert(0, norm_path)
        groups.insert(0, "terminal_norm")

    names = tuple(model.named_parameters())
    selected_names = tuple(
        name
        for name, _parameter in names
        if any(_has_prefix(name, prefix) for prefix in prefixes)
    )
    if not selected_names:
        raise RuntimeError(
            f"SenseNova structured policy selected no parameters; prefixes={prefixes}"
        )
    selected_ids = set(selected_names)
    for name, parameter in names:
        parameter.requires_grad_(name in selected_ids)
    return SenseNovaPolicySelection(
        policy=policy,
        groups=tuple(groups),
        parameter_names=selected_names,
    )
