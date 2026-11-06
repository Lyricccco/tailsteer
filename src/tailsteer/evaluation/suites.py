"""Small evaluation profiles shared by standalone and post-training evaluation."""

import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from .common import BENCHMARKS, DEFAULT_ROOT, PROJECT, atomic_text, digest
from .judge_config import load_judge_config

DEFAULT_INFERENCE_SUITES = ("cafe-20turn",)


def inference_suites(value) -> tuple[str, ...]:
    """Accept a single suite or an ordered, repeated CLI selection."""
    values = (
        DEFAULT_INFERENCE_SUITES
        if value is None
        else [value]
        if isinstance(value, (str, Path))
        else value
    )
    if not values or any(
        not isinstance(v, (str, Path)) or not str(v).strip() for v in values
    ):
        raise ValueError("inference_suite must not be empty")
    result = tuple(str(v).strip() for v in values)
    if len(result) != len(set(result)):
        raise ValueError("inference suites must be unique")
    return result


def _path(value) -> Path:
    path = Path(str(value)).expanduser()
    return (PROJECT / path).resolve()


def evaluation_suite(value: str | Path) -> Path | None:
    """Native aliases/YAML remain native; only kind=evaluation is dispatched."""
    path = _path(value)
    if not path.is_file():
        if path.suffix in {".yaml", ".yml"}:
            raise FileNotFoundError(path)
        return None
    config = yaml.safe_load(path.read_text())
    return (
        path
        if isinstance(config, dict) and config.get("kind") == "evaluation"
        else None
    )


def load_suite(path: str | Path) -> dict:
    config = yaml.safe_load(_path(path).read_text())
    allowed = {
        "kind",
        "benchmarks",
        "policies",
        "endpoint_modes",
        "feedback",
        "seed",
        "limit",
        "stage",
        "eval_root",
        "workers",
        "judge_config",
        "api",
        "gedit_language",
        "input_max_pixels",
    }
    if not isinstance(config, dict) or config.get("kind") != "evaluation":
        raise ValueError("Evaluation suite requires kind: evaluation")
    if config.keys() - allowed:
        raise ValueError(
            f"Unknown evaluation suite fields: {sorted(config.keys() - allowed)}"
        )
    for key in ("benchmarks", "policies", "endpoint_modes"):
        values = config.get(key)
        if values is None and key != "benchmarks":
            continue
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or not v for v in values)
        ):
            raise ValueError(f"{key} must be a nonempty string list")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {key}")
    if set(config["benchmarks"]) - set(BENCHMARKS):
        raise ValueError("Unknown evaluation benchmark")
    if config.get("policies") and config.get("endpoint_modes"):
        raise ValueError("Select policies or endpoint_modes, not both")
    for key, choices, default in (
        ("stage", {"generate", "score", "report", "all"}, "all"),
        ("feedback", {"image", "latent", "continuous-image", "model-native"}, "image"),
        ("gedit_language", {"all", "en", "cn"}, "all"),
    ):
        if config.get(key, default) not in choices:
            raise ValueError(f"Invalid evaluation {key}")
        config.setdefault(key, default)
    for key in ("seed", "limit", "workers", "input_max_pixels"):
        value = config.get(key)
        if value is not None and (
            type(value) is not int or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"Invalid evaluation {key}")
    config["eval_root"] = str(_path(config.get("eval_root", DEFAULT_ROOT)))
    if config.get("judge_config"):
        config["judge_config"] = str(_path(config["judge_config"]))
        load_judge_config(Path(config["judge_config"]))
    api = config.get("api", {})
    if not isinstance(api, dict) or api.keys() - {"key_file", "base_url"}:
        raise ValueError(
            "Evaluation api accepts only key_file and base_url, never a raw key"
        )
    if api.get("key_file"):
        api["key_file"] = str(_path(api["key_file"]))
    if api.get("base_url"):
        url = urlsplit(api["base_url"])
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Evaluation API URL must not contain credentials/query/fragment"
            )
    config["api"] = api
    return config


def suite_defaults(config: dict) -> dict:
    rename = {
        "benchmarks": "bench",
        "policies": "policy",
        "endpoint_modes": "endpoint_mode",
    }
    defaults = {
        rename.get(k, k): v for k, v in config.items() if k not in {"kind", "api"}
    }
    defaults.update({f"api_{k}": v for k, v in config.get("api", {}).items()})
    return defaults


def check_suite_assets(config: dict, *, gpus: int) -> None:
    """Cheap local preflight; no model load, API call or credential read."""
    root = Path(config["eval_root"])
    for bench in config["benchmarks"]:
        manifest = root / "prepared" / bench / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    if config["stage"] not in {"all", "score"}:
        return
    for env in {
        "edival" if b == "edival" else "api"
        for b in config["benchmarks"]
        if b != "qwenimagebench"
    }:
        python = root / "envs" / env / "bin/python"
        if not python.is_file() or not os.access(python, os.X_OK):
            raise FileNotFoundError(f"Evaluation environment is unavailable: {python}")
    if "qwenimagebench" in config["benchmarks"]:
        from .qwenimagebench import settings

        judge = settings()
        for path in (
            Path(judge["python"]),
            Path(judge["model"]) / "model.safetensors.index.json",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
    if set(config["benchmarks"]) - {"edival", "qwenimagebench"}:
        key_file = config.get("api", {}).get("key_file") or os.environ.get(
            "OPENAI_API_KEY_FILE"
        )
        if key_file:
            if not Path(key_file).is_file() or not os.access(key_file, os.R_OK):
                raise FileNotFoundError("Evaluation API key file is not readable")
        elif not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "Evaluation API key file or environment credential is required"
            )
    if "edival" in config["benchmarks"]:
        if gpus < 2:
            raise ValueError(
                "EdiVal's native GPU judge requires at least two visible GPUs"
            )
        from .judge_assets import verify_edival_judges

        verify_edival_judges(root)


def snapshot_suite(value: str, directory: Path, *, gpus: int, check_assets=True) -> str:
    """Freeze suite and judge YAML together, without copying credentials."""
    path = evaluation_suite(value)
    if path is None:
        return value
    config = load_suite(path)
    if check_assets:
        check_suite_assets(config, gpus=gpus)
    if config.get("judge_config"):
        judges = yaml.safe_load(Path(config["judge_config"]).read_text())
        judge_path = directory / f"judges-{digest(judges)[:12]}.yaml"
        atomic_text(judge_path, yaml.safe_dump(judges, sort_keys=False))
        config["judge_config"] = str(judge_path.resolve())
    destination = directory / f"evaluation-{digest(config)[:12]}.yaml"
    atomic_text(destination, yaml.safe_dump(config, sort_keys=False))
    return str(destination.resolve())
