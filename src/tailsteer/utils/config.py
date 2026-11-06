"""Complete-YAML configuration and shared storage-root handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / "cache"
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[3] / "models"
DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[3] / "runs"


def workspace_root() -> Path:
    """Return the code workspace, with one optional environment override."""

    configured = os.environ.get("TAILSTEER_ROOT")
    if configured:
        root = Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
    else:
        # Repository root in an editable installation.
        root = Path(__file__).resolve().parents[3]
    if not root.is_dir():
        raise FileNotFoundError(f"TailSteer repository does not exist: {root}")
    return root


def _shared_root(environment: str, default: Path) -> Path:
    configured = os.environ.get(environment)
    value = configured if configured else str(default)
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def data_root() -> Path:
    """Return the reusable input-dataset root."""

    return _shared_root("TAILSTEER_DATA_ROOT", DEFAULT_DATA_ROOT)


def cache_root() -> Path:
    """Return the generated and reusable cache root."""

    return _shared_root("TAILSTEER_CACHE_ROOT", DEFAULT_CACHE_ROOT)


def model_root() -> Path:
    """Return the shared immutable model-snapshot root."""

    return _shared_root("TAILSTEER_MODEL_ROOT", DEFAULT_MODEL_ROOT)


def run_root() -> Path:
    """Return the shared experiment-output root."""

    return _shared_root("TAILSTEER_RUN_ROOT", DEFAULT_RUN_ROOT)


def _expand(
    value: Any,
    *,
    workspace: Path,
    datasets: Path,
    caches: Path,
    models: Path,
    runs: Path,
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _expand(
                item,
                workspace=workspace,
                datasets=datasets,
                caches=caches,
                models=models,
                runs=runs,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _expand(
                item,
                workspace=workspace,
                datasets=datasets,
                caches=caches,
                models=models,
                runs=runs,
            )
            for item in value
        ]
    if isinstance(value, str):
        # Expand every supported root even when the caller did not export it.
        # Other environment variables retain normal shell semantics.
        text = value.replace("${TAILSTEER_ROOT}", str(workspace))
        text = text.replace("${TAILSTEER_DATA_ROOT}", str(datasets))
        text = text.replace("${TAILSTEER_CACHE_ROOT}", str(caches))
        text = text.replace("${TAILSTEER_MODEL_ROOT}", str(models))
        text = text.replace("${TAILSTEER_RUN_ROOT}", str(runs))
        return os.path.expandvars(os.path.expanduser(text))
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a YAML mapping: {source}")
    resolved = _expand(
        value,
        workspace=workspace_root(),
        datasets=data_root(),
        caches=cache_root(),
        models=model_root(),
        runs=run_root(),
    )
    from tailsteer.configuration import normalize

    resolved = normalize(resolved)
    resolved["_config_path"] = str(source)
    return resolved


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir).expanduser().resolve() / "config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
