"""Versioned format-only adaptation of ImgEdit Basic's three scoring fields."""

import re
from pathlib import Path

from .common import read_json

PARSER_VERSION = "strict_fields_v1"


def scoring_fields(prompt: str) -> tuple[str, ...]:
    """Use the pinned official response template, not a second rubric."""
    fields = tuple(re.findall(r"^([^\n:]+): A number from 1 to 5\.$", prompt, re.M))
    if len(fields) != 3 or len(set(fields)) != 3:
        raise ValueError("Expected three distinct scoring fields in official prompt")
    return fields


def normalize_response(response: str, fields: tuple[str, ...]) -> dict:
    """Accept decorations/explanations, never infer, clip or fill missing scores."""
    if not isinstance(response, str):
        raise ValueError("ImgEdit response must be text")
    labels = {field.casefold(): field for field in fields}
    scores = {}
    for line in response.splitlines():
        # These decorations are discarded only for identifying score lines.
        for marker in ("**", "__", "`", "*"):
            line = line.replace(marker, "")
        line = re.sub(r"^\s*(?:(?:#{1,6}|[-+])\s+|\d+[.)]\s+)+", "", line)
        label, separator, value = line.partition(":")
        field = labels.get(" ".join(label.split()).casefold())
        if not separator or field is None:
            continue
        if field in scores:
            raise ValueError(f"Duplicate ImgEdit scoring field: {field}")
        # Integer 1..5, optionally followed by a parenthesized/dash explanation.
        # A closing parenthesis may itself be followed by sentence punctuation.
        # Decimal scores, fractions, ranges and bare alternative values fail.
        match = re.fullmatch(
            r"([1-5])(?:\s+\(.*\)[.!]?|\s+[-–—]\s+(?![+\-\d])\S.*|[.!]\s+\D.*)?",
            value.strip(),
        )
        if match is None:
            raise ValueError(f"Invalid ImgEdit score for {field}: {value.strip()}")
        scores[field] = int(match[1])
    missing = set(fields) - scores.keys()
    if missing:
        raise ValueError(f"Missing ImgEdit scoring fields: {sorted(missing)}")
    return {
        "parser": PARSER_VERSION,
        "response": response,
        "normalized": "\n".join(f"{field}: {scores[field]}" for field in fields),
        "scores": {field: scores[field] for field in fields},
    }


def response_directories(work: Path, bundle: dict) -> list[Path]:
    """Replay old raw responses only when all non-parser scorer inputs match."""
    directories = [work / "responses"]
    job = read_json(Path(bundle["job"]))
    source_job = job.get("response_source")
    if not source_job:
        return directories
    source = Path(read_json(Path(source_job))["output"]) / "scores" / work.name
    if not (source / "scorer.json").is_file() or not (work / "scorer.json").is_file():
        return directories
    old, new = (read_json(p / "scorer.json") for p in (source, work))
    if old.get("parser", "official") != "official":
        return directories
    for identity in (old, new):
        identity.pop("parser", None)
        identity.setdefault("judge_api", "chat")
    if old == new:
        directories.insert(0, source / "responses")
    return directories


def replay_response(row_id: str, fields: tuple[str, ...], directories: list[Path]):
    """First valid response in saved request order, regardless of score."""
    paths = sorted(
        (
            path
            for directory in directories
            for path in directory.glob(f"{row_id}-*.json")
        ),
        key=lambda path: (path.name, str(path)),
    )
    for path in paths:
        try:
            result = normalize_response(read_json(path).get("response"), fields)
        except ValueError:
            continue
        return {**result, "source_response": str(path.resolve())}
    return None
