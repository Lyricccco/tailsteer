"""Single-copy pretrained/refined switching for full transformer exports."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any


class _ActorState:
    actor = "base"


class _ActorParameter:
    @staticmethod
    def build(refined: Any, state: _ActorState) -> Any:
        import torch

        class _Parametrization(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("refined", refined)
                self.state = state

            def forward(self, pretrained: Any) -> Any:
                return self.refined if self.state.actor == "refined" else pretrained

        return _Parametrization()


class TransformerActors:
    """Switch one official transformer once per scheduler interval."""

    def __init__(
        self,
        transformer: Any,
        export_path: str | Path,
        *,
        model_name: str,
        export_format: str,
        offload_weights: bool = False,
    ) -> None:
        from ..checkpoint import checkpoint_manifest, open_weights
        from torch.nn.utils import parametrize

        self.state = _ActorState()
        self._registered = []
        self._offloaded = {}
        self._parameters = {}
        self.manifest = {}
        if export_path is not None:
            root = Path(export_path).expanduser().resolve()
            if (
                not (root / "tailrefine_release.json").is_file()
                and not (root / "COMPLETE").is_file()
            ):
                raise RuntimeError(f"incomplete transformer export: {root}")
            self.manifest = checkpoint_manifest(root)
            if self.manifest.get("model") != model_name:
                raise ValueError("transformer export model differs from runtime")
            if (
                self.manifest.get("format", "").replace("tailrefine", "tailsteer")
                != export_format
            ):
                raise ValueError("transformer export format differs from runtime")
            if self.manifest.get("weight_dtype") != "bfloat16":
                raise ValueError("transformer actor export must use BF16 weights")
            weights = root / str(self.manifest.get("weights", ""))
            if not weights.is_file():
                raise FileNotFoundError(weights)

            transformer.requires_grad_(False)
            parameters = dict(transformer.named_parameters())
            self.state = _ActorState()
            self._registered: list[tuple[Any, str]] = []
            self._offloaded: dict[str, dict[str, Any]] = {}
            self._parameters = parameters if offload_weights else {}
            if offload_weights:
                self._offloaded = {
                    "base": {
                        name: value.detach().cpu().clone()
                        for name, value in parameters.items()
                    },
                    "refined": {},
                }
            with open_weights(weights) as handle:
                names = tuple(handle.keys())
                if set(names) != set(parameters):
                    missing = sorted(set(parameters) - set(names))
                    extra = sorted(set(names) - set(parameters))
                    raise RuntimeError(
                        "transformer export parameter set differs from runtime: "
                        f"missing={missing[:3]} extra={extra[:3]}"
                    )
                for name in names:
                    pretrained = parameters[name]
                    refined = handle.get_tensor(name).to(
                        device="cpu" if offload_weights else pretrained.device,
                        dtype=pretrained.dtype,
                    )
                    if refined.shape != pretrained.shape:
                        raise ValueError(f"transformer export shape mismatch: {name}")
                    parent_name, _, attribute = name.rpartition(".")
                    parent = (
                        transformer.get_submodule(parent_name)
                        if parent_name
                        else transformer
                    )
                    if offload_weights:
                        self._offloaded["refined"][name] = refined
                    else:
                        parametrize.register_parametrization(
                            parent,
                            attribute,
                            _ActorParameter.build(refined, self.state),
                            unsafe=True,
                        )
                    self._registered.append((parent, attribute))
            if not self._registered:
                raise RuntimeError("transformer actor export has no parameters")
        self._active = False
        self._sequence: tuple[str, ...] = ()
        self._index = 0
        self._trace: list[str] = []

    def _activate(self, actor: str) -> None:
        if self._offloaded and actor != self.state.actor:
            import torch

            # One GPU copy, transfers only at policy boundaries, not per layer
            # or forward. Default parametrized actors keep their old behavior.
            with torch.no_grad():
                for name, parameter in self._parameters.items():
                    parameter.copy_(self._offloaded[actor][name])
        self.state.actor = actor

    @property
    def parameter_count(self) -> int:
        return len(self._registered)

    @contextmanager
    def use_policy(self, policy: Any) -> Any:
        if self._active:
            raise RuntimeError("nested transformer actor policies are unsupported")
        self._sequence = tuple(
            policy.actor_at(index) for index in range(policy.num_steps)
        )
        self._index = 0
        self._trace = []
        self._active = True
        self._activate(self._sequence[0])
        try:
            yield
            if self._index != len(self._sequence):
                raise RuntimeError(
                    "sampler actor-step count mismatch: "
                    f"expected={len(self._sequence)} actual={self._index}"
                )
        finally:
            self._active = False
            self._activate("base")

    def complete_step(self) -> None:
        if not self._active or self._index >= len(self._sequence):
            raise RuntimeError("sampler completed an unexpected actor step")
        actor = self._sequence[self._index]
        if self.state.actor != actor:
            raise RuntimeError("transformer actor changed within a scheduler step")
        self._trace.append(actor)
        self._index += 1
        if self._index < len(self._sequence):
            self._activate(self._sequence[self._index])

    @property
    def trace(self) -> tuple[str, ...]:
        return tuple(self._trace)
