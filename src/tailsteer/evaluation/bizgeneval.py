"""BizGenEval official prompts and scoring rules, with strict resumable coverage.

Only prompts, image transport and metric helpers are imported from the pinned
snapshot. Missing judge answers are failures, never silently negative scores.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

from .common import digest, read_json, write_json


def image_size(ratio: str, long_side: int = 1024) -> tuple[int, int]:
    """Aspect-ratio/long-side rule, rounded to the native 32px grid."""
    ratio = (ratio or "").strip() or "1:1"
    w, h = (int(v) for v in ratio.lower().replace("x", ":").split(":"))
    if min(w, h) <= 0:
        raise ValueError("BizGenEval aspect ratio must be positive")
    return tuple(max(32, round(long_side * v / max(w, h) / 32) * 32) for v in (w, h))


def prepare(root: Path, source: Path | None = None) -> Path:
    from .sources import prepare_code

    code = prepare_code(root, "bizgeneval")
    raw = source / "test.jsonl" if source else code / "assets/bizgeneval.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    if len(rows) != 400 or len({r["id"] for r in rows}) != 400:
        raise ValueError("BizGenEval requires 400 unique prompts")
    upstream = _upstream(code)
    records = []
    for i, row in enumerate(rows):
        tag = row.get("eval_tag") or row["dimension"]
        if tag not in upstream.EVAL_PROMPTS or not row.get("questions"):
            raise ValueError(f"Invalid BizGenEval checklist: {i}")
        width, height = image_size(row["aspect_ratio"])
        records.append(
            {
                "id": f"bizgen-{i:04d}",
                "official_id": row["id"],
                "prompts": [row["prompt"]],
                "width": width,
                "height": height,
                "metadata": {**row, "ID": i + 1, "prompt_id": i},
            }
        )
    manifest = {
        "schema_version": 1,
        "benchmark": "bizgeneval",
        "task_type": "t2i",
        "records": records,
        "fingerprint": digest(records),
        "sessions": 400,
        "turns": 400,
        "raw_source": str(raw),
        "official_code": str(code),
    }
    path = root / "prepared/bizgeneval/manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError(f"Prepared BizGenEval identity changed: {path}")
    write_json(path, manifest)
    return path


def _upstream(code: Path):
    # Load only official prompt/parser definitions, without executing its CLI.
    import runpy
    from types import SimpleNamespace
    from .bizgen_metrics import (
        encode_image,
        item_scores,
        aggregate,
        render_prompt,
        to_bool,
    )

    prompts = runpy.run_path(str(code / "evaluation/evaluate_prompt.py"))[
        "EVAL_GENERATION_PROMPTS"
    ]
    parser = runpy.run_path(str(code / "utils/parsers.py"))["parse_json_safe"]
    return SimpleNamespace(
        EVAL_PROMPTS=prompts,
        _parse_json_safe=parser,
        _extract_results_only=lambda raw, n: None,
        _to_bool=to_bool,
        _render_prompt=render_prompt,
        _encode_image=encode_image,
        _compute_item_scores=item_scores,
        _aggregate=aggregate,
        ERROR_ALPHA=0.1,
    )


def parse_answers(raw: str, questions: list[str], upstream) -> dict:
    from .api_retry import InvalidJudgement

    try:
        answers = upstream._parse_json_safe(raw)
    except (ValueError, SyntaxError, TypeError):
        answers = upstream._extract_results_only(raw, len(questions))
    expected = {str(i) for i in range(1, len(questions) + 1)}
    if not isinstance(answers, dict) or set(answers) != expected:
        raise InvalidJudgement("BizGenEval returned an incomplete checklist")
    meta = {}
    for i, question in enumerate(questions, 1):
        entry = answers[str(i)]
        value = (
            upstream._to_bool(entry.get("result")) if isinstance(entry, dict) else None
        )
        if value is None:
            raise InvalidJudgement(f"Invalid BizGenEval answer {i}")
        meta[str(i)] = {
            "result": value,
            "raw_description": question,
            "reason": entry.get("reason", ""),
        }
    return meta


def score(code: Path, bundle: dict, work: Path, workers: int) -> dict:
    from openai import OpenAI

    from .api_retry import JudgeAPI
    from .judge_config import JUDGES
    from .scorers import cached_items

    upstream = _upstream(code)
    client = JudgeAPI(work, bundle).client(OpenAI)

    def evaluate(row):
        item = row["metadata"]
        system, template = upstream.EVAL_PROMPTS[
            item.get("eval_tag") or item["dimension"]
        ]
        response = client.chat.completions.create(
            model=JUDGES["bizgeneval"],
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": upstream._render_prompt(
                                template, item["questions"]
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": upstream._encode_image(Path(row["outputs"][0]))
                            },
                        },
                    ],
                },
            ],
        )
        raw = response.choices[0].message.content
        write_json(
            work / "responses" / f"{row['id']}-{time.time_ns()}.json",
            {"response": raw, "model": response.model},
        )
        meta = parse_answers(raw, item["questions"], upstream)
        overall = upstream._compute_item_scores(item, meta, upstream.ERROR_ALPHA)
        result = {
            "prompt_id": item["prompt_id"],
            "dataset_id": item["id"],
            "domain": item["domain"],
            "dimension": item["dimension"],
            **overall,
            "meta_info": meta,
            "raw_model_response": raw,
        }
        for name in ("easy", "hard"):
            scores = upstream._compute_item_scores(
                item, meta, upstream.ERROR_ALPHA * 2, f"{name}_qidxs"
            )
            result.update(
                {f"{name}_{k}": scores[k] for k in ("accuracy", "error_score")}
            )
        return result

    values = cached_items(bundle["records"], work, evaluate, workers)
    records = [values[row["id"]] for row in bundle["records"]]
    metrics = {
        "items": len(records),
        "judge_model": JUDGES["bizgeneval"],
        "error_alpha": upstream.ERROR_ALPHA,
        "easy_hard_error_alpha": upstream.ERROR_ALPHA * 2,
        "by_domain": upstream._aggregate(records, "domain"),
        "by_dimension": upstream._aggregate(records, "dimension"),
        "note": (
            "Official BizGenEval prompts and error-score aggregation; Gemini-3-Flash-Preview judge; "
            "native aspect ratio, 1024px long side; JPEG90 judge transport; "
            "strict complete checklists. No missing answers counted as false."
        ),
    }
    for key in (
        "accuracy",
        "error_score",
        "easy_accuracy",
        "easy_error_score",
        "hard_accuracy",
        "hard_error_score",
    ):
        metrics[("overall_" + key) if key in {"accuracy", "error_score"} else key] = (
            mean(r[key] for r in records)
        )
    write_json(work / "bizgeneval_summary.json", {**metrics, "records": records})
    return metrics
