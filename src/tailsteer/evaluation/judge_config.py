"""Explicit judge variants, independent of image-generation settings."""

from pathlib import Path

from .imgedit_parser import PARSER_VERSION

JUDGES = {
    "gedit": "gpt-4.1",
    "rise": "gpt-4.1-2025-04-14",
    "imgedit-basic": "gpt-4o",
    "mse": "gpt-5-nano",
    "edival": "official-local-detectors",
    "qwenimagebench": "Q-Judger",
    "bizgeneval": "gemini-3-flash-preview",
}


def judge_settings(benchmark: str) -> dict:
    return {
        "model": JUDGES[benchmark],
        "api": "responses"
        if benchmark == "mse"
        else "local"
        if benchmark in {"edival", "qwenimagebench"}
        else "chat",
    }


def add_judge_argument(parser):
    parser.add_argument(
        "--judge-config",
        type=Path,
        help="Optional scoring-only YAML (RISE model / MSE API / ImgEdit parser).",
    )


def load_judge_config(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    import yaml

    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Judge config must be a benchmark-to-settings mapping")
    supported = {"rise": {"model"}, "mse": {"api"}, "imgedit-basic": {"parser"}}
    resolved = {}
    for bench, overrides in config.items():
        if (
            bench not in supported
            or not isinstance(overrides, dict)
            or overrides.keys() - supported[bench]
        ):
            raise ValueError(f"Unsupported judge override: {bench}")
        settings = {**judge_settings(bench), **overrides}
        if not isinstance(settings["model"], str) or not settings["model"].strip():
            raise ValueError("Judge model must be a nonempty string")
        if settings["api"] not in {"chat", "responses"}:
            raise ValueError("MSE API must be chat or responses")
        if bench == "imgedit-basic":
            parser_name = settings.get("parser", "official")
            if not isinstance(parser_name, str) or parser_name not in {
                "official",
                PARSER_VERSION,
            }:
                raise ValueError(f"ImgEdit parser must be official or {PARSER_VERSION}")
            if settings.get("parser") == "official":
                settings.pop("parser")
        if settings != judge_settings(bench):
            resolved[bench] = settings
    return resolved
