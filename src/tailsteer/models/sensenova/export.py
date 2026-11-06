"""Validated metadata for portable SenseNova inference exports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_TRAINABLE = {
    "policy": "generation_denoiser",
    "master_dtype": "float32",
    "reference_vision_model": False,
    "prefix_kv_adapter": {"enabled": False},
    "conditioning_gradients": False,
}


@dataclass(frozen=True)
class SenseNovaExportSpec:
    """One normalized schema-v1/v2/v3 SenseNova delta export."""

    root: Path
    manifest: dict[str, Any]
    schema_version: int
    trainable: dict[str, Any]
    weights: Path

    @classmethod
    def load(cls, export_path: str | Path) -> SenseNovaExportSpec:
        root = Path(export_path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not (root / "tailrefine_release.json").is_file() and (
            not (root / "COMPLETE").is_file() or not manifest_path.is_file()
        ):
            raise RuntimeError(f"incomplete SenseNova inference export: {root}")
        from ..checkpoint import checkpoint_manifest

        manifest = checkpoint_manifest(root)
        if not isinstance(manifest, dict):
            raise TypeError(f"SenseNova export manifest must be a mapping: {root}")
        export_format = manifest.get("format")
        if isinstance(export_format, str):
            export_format = export_format.replace("tailrefine", "tailsteer")
        if export_format not in {None, "tailsteer_sensenova_delta"}:
            raise ValueError(f"not a SenseNova inference delta: {manifest_path}")
        schema_version = int(manifest.get("schema_version", 1))
        if schema_version not in {1, 2, 3}:
            raise ValueError(
                f"unsupported SenseNova export schema {schema_version}: {manifest_path}"
            )
        raw_trainable = manifest.get("trainable")
        if raw_trainable is None:
            trainable = deepcopy(_DEFAULT_TRAINABLE)
            trainable["policy"] = str(
                manifest.get("trainable_policy", "generation_denoiser")
            )
        elif isinstance(raw_trainable, dict):
            trainable = deepcopy(raw_trainable)
        else:
            raise TypeError("SenseNova export trainable config must be a mapping")
        trainable.setdefault(
            "policy", str(manifest.get("trainable_policy", "generation_denoiser"))
        )
        trainable.setdefault("master_dtype", "float32")
        trainable.setdefault("reference_vision_model", False)
        trainable.setdefault("prefix_kv_adapter", {"enabled": False})
        trainable.setdefault(
            "conditioning_gradients", trainable["policy"] == "full_model"
        )
        if schema_version == 1 and (
            bool(trainable["reference_vision_model"])
            or bool(dict(trainable["prefix_kv_adapter"]).get("enabled", False))
            or bool(trainable["conditioning_gradients"])
        ):
            raise ValueError("SenseNova schema-v1 exports cannot train conditioning")
        if trainable["policy"] == "full_model" and schema_version < 3:
            raise ValueError("SenseNova full_model exports require schema v3")
        raw_weights = manifest.get("weights")
        if not isinstance(raw_weights, str) or not raw_weights:
            raise ValueError(f"SenseNova export has no weights file: {manifest_path}")
        weights = root / raw_weights
        if not weights.is_file():
            raise FileNotFoundError(f"SenseNova export weights do not exist: {weights}")
        return cls(
            root=root,
            manifest=manifest,
            schema_version=schema_version,
            trainable=trainable,
            weights=weights,
        )

    @property
    def has_reference_actor(self) -> bool:
        return self.trainable.get("policy") == "full_model" or bool(
            self.trainable.get("reference_vision_model", False)
        )

    @property
    def has_prefix_adapter(self) -> bool:
        raw = self.trainable.get("prefix_kv_adapter", {})
        return isinstance(raw, dict) and bool(raw.get("enabled", False))
