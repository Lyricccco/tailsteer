"""Inference backend lookup, separate from model construction registration."""

from __future__ import annotations

from typing import Any

from .backends.base import InferenceBackend
from .backends.flux2 import Flux2InferenceBackend
from .backends.native_transformer import (
    KontextInferenceBackend,
    QwenImage21InferenceBackend,
)
from .backends.sensenova import SenseNovaInferenceBackend

_BACKENDS: tuple[InferenceBackend, ...] = (
    Flux2InferenceBackend(),
    KontextInferenceBackend(),
    QwenImage21InferenceBackend(),
    SenseNovaInferenceBackend(),
)

INFERENCE_BACKENDS = {backend.key: backend for backend in _BACKENDS}
MODEL_INFERENCE_BACKENDS = {
    model_name: backend for backend in _BACKENDS for model_name in backend.model_names
}


def inference_backend_for_model(model_name: str) -> InferenceBackend:
    try:
        return MODEL_INFERENCE_BACKENDS[str(model_name)]
    except KeyError as exc:
        raise ValueError(
            f"model {model_name!r} has no inference backend; "
            f"available={sorted(MODEL_INFERENCE_BACKENDS)}"
        ) from exc


def run_registered_inference(
    config: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, dict):
        raise TypeError("inference config requires a model mapping")
    backend = inference_backend_for_model(str(model.get("name", "")))
    evaluation = config.get("evaluation", {})
    configured = evaluation.get("backend") if isinstance(evaluation, dict) else None
    if configured is not None and str(configured) != backend.key:
        raise ValueError(
            f"inference backend mismatch: config={configured!r} model={backend.key!r}"
        )
    return backend.run(config, resume=resume)
