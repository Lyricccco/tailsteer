"""Thin official-scorer bindings with explicit optional judge/transport variants.

Each benchmark executes in its own process, so upstream modules called `utils`
cannot collide. Prompts, image transforms and aggregation stay upstream; parser
compatibility is explicit, versioned and isolated from native results.
"""

from __future__ import annotations

import csv
import importlib
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from .api import mse_responses_via_chat, redact_secret
from .api_retry import (
    IncompleteJudgements,
    InvalidJudgement,
    JudgeAPI,
    JudgeAPIError,
)
from .common import digest, read_json, write_json
from .imgedit_parser import (
    PARSER_VERSION,
    normalize_response,
    replay_response,
    response_directories,
    scoring_fields,
)
from .judge_config import JUDGES as JUDGES
from .judge_config import judge_settings


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def official_module(code: Path, name: str):
    sys.path.insert(0, str(code))
    return importlib.import_module(name)


def cached_items(rows, work: Path, evaluate, workers: int):
    """Cache successes and retain errors; retry failed items on the next call."""

    def one(row):
        path = work / "items" / f"{row['id']}.json"
        if path.exists() and read_json(path).get("status") == "complete":
            return read_json(path)
        try:
            value = evaluate(row)
            if value is None:
                raise ValueError("Official scorer returned no result")
            result = {"status": "complete", "id": row["id"], "result": value}
        except Exception as error:
            result = {
                "status": "failed",
                "id": row["id"],
                "error": redact_secret(str(error)),
                "retryable": isinstance(error, InvalidJudgement)
                or (isinstance(error, JudgeAPIError) and error.retryable),
            }
        write_json(path, finite(result))
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(one, rows))
    failed = [r for r in results if r["status"] != "complete"]
    write_json(
        work / "coverage.json",
        {"expected": len(rows), "complete": len(rows) - len(failed), "failed": failed},
    )
    if failed:
        raise IncompleteJudgements(
            f"{len(failed)}/{len(rows)} judgements failed; "
            f"see {work / 'coverage.json'}, rerun --resume",
            retryable=any(r.get("retryable", False) for r in failed),
        )
    return {r["id"]: r["result"] for r in results}


def imgedit(code: Path, bundle: dict, work: Path, workers: int) -> dict:
    from openai import OpenAI

    module = official_module(code, "basic_bench")
    transport = JudgeAPI(work, bundle)
    # Upstream has literal api-key/url placeholders. Keep its request unchanged.
    module.OpenAI = lambda **ignored: transport.client(OpenAI)
    prompts = module.load_prompts(str(code / "prompts.json"))
    parser = official_module(code, "step1_get_avgscore")
    aggregate = official_module(code, "step2_typescore")
    parser_name = bundle.get("judge", {}).get("parser", "official")
    if parser_name not in {"official", PARSER_VERSION}:
        raise ValueError(f"Unsupported ImgEdit parser: {parser_name}")
    strict = parser_name == PARSER_VERSION
    fields = (
        {kind: scoring_fields(prompt) for kind, prompt in prompts.items()}
        if strict
        else {}
    )
    response_dirs = response_directories(work, bundle) if strict else []

    def evaluate(row):
        meta = row["metadata"]
        if strict:
            replayed = replay_response(
                row["id"], fields[meta["edit_type"]], response_dirs
            )
            if replayed is not None:
                return replayed
        # Retry invalid formatting, never low scores; keep the request unchanged
        # and retain every response, including those rejected by the parser.
        for _ in range(3):
            _, response = module.process_single_item(
                meta["key"], meta, str(work / "images"), str(work / "sources"), prompts
            )
            path = work / "responses" / f"{row['id']}-{time.time_ns()}.json"
            if strict:
                try:
                    result = normalize_response(response, fields[meta["edit_type"]])
                except ValueError as error:
                    write_json(
                        path,
                        {
                            "response": response,
                            "parser": parser_name,
                            "error": str(error),
                        },
                    )
                    continue
                write_json(path, result)
                return {**result, "source_response": str(path.resolve())}
            write_json(path, {"response": response})
            if parser.extract_scores_and_average(response) is not None:
                return response
        raise InvalidJudgement(
            "Unparseable official response after 3 attempts; see responses/"
        )

    results = cached_items(bundle["records"], work, evaluate, workers)
    raw = {r["metadata"]["key"]: results[r["id"]] for r in bundle["records"]}
    if strict:
        normalized = {key: result["normalized"] for key, result in raw.items()}
        raw = {key: result["response"] for key, result in raw.items()}
        write_json(work / "normalized_result.json", normalized)
    else:
        normalized = raw
    scores = parser.compute_averages(normalized)
    by_type = aggregate.compute_edit_type_averages(
        scores, {r["metadata"]["key"]: r["metadata"] for r in bundle["records"]}
    )
    write_json(work / "result.json", raw)
    write_json(work / "average_scores.json", scores)
    write_json(work / "typescore.json", by_type)
    return {"category_macro": sum(by_type.values()) / len(by_type), "by_type": by_type}


def gedit(code: Path, bundle: dict, work: Path, workers: int) -> dict:
    from PIL import Image

    sys.path.insert(0, str(code / "viescore"))
    module = official_module(code, "run_gedit_score")
    parse_response = importlib.import_module("viescore").mllm_output_to_dict
    api = importlib.import_module("mllm_tools.openai")
    api.get_api_key = lambda ignored: os.environ["OPENAI_API_KEY"]
    # Retain upstream requests/payloads; supply our endpoint and a finite timeout.
    import requests

    transport = JudgeAPI(work, bundle)
    api.requests = SimpleNamespace(post=transport.post, get=requests.get)

    def evaluate(row):
        scorer = module.VIEScore(backbone="gpt4o", task="tie", key_path=None)
        scorer.model.url = (
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            + "/chat/completions"
        )
        raw = []
        errors = []
        original = scorer.model.get_parsed_output

        def capture(*args, **kwargs):
            try:
                response = original(*args, **kwargs)
            except JudgeAPIError as error:
                errors.append(error)
                raise
            raw.append(response)
            write_json(
                work / "responses" / f"{row['id']}-{time.time_ns()}.json",
                {"response": response},
            )
            if not response or response == "rate_limit_exceeded":
                raise InvalidJudgement("GEdit API returned an empty/error response")
            # VIEScore otherwise guesses a random score after parse failures.
            # Accept its native parser, but never turn an invalid reply into a
            # fabricated score. The outer bounded loop retries this item.
            parsed = parse_response(response, give_up_parsing=False)
            if not isinstance(parsed, dict) or not parsed.get("score"):
                raise InvalidJudgement("Unparseable GEdit response; see responses/")
            return response

        scorer.model.get_parsed_output = capture
        item = {
            **row["metadata"],
            "input_image_raw": Image.open(row["source"]),
            "edited_image_path": row["outputs"][0],
        }
        try:
            result = module.process_single_item(item, scorer, max_retries=2)
        except NameError as error:
            # Upstream's exhausted-retry diagnostic references an undefined
            # save_path_item. Preserve the actual transport/parse failure.
            if error.name != "save_path_item":
                raise
            if errors:
                raise errors[-1] from error
            raise InvalidJudgement(
                "GEdit could not parse a judgement; see responses/"
            ) from error
        if result is None:
            raise ValueError("GEdit returned no score")
        return {"scores": result, "responses": raw}

    results = cached_items(bundle["records"], work, evaluate, workers)
    stats = official_module(code, "calculate_statistics")
    fields = [
        "key",
        "edited_image",
        "instruction",
        "sementics_score",
        "quality_score",
        "intersection_exist",
        "instruction_language",
    ]
    for group in stats.GROUPS:
        with (work / f"model_{group}_all_vie_score.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                results[r["id"]]["scores"]
                for r in bundle["records"]
                if r["metadata"]["task_type"] == group
            )
    summary = {}
    for language in sorted(
        {r["metadata"]["instruction_language"] for r in bundle["records"]}
    ):
        values = stats.analyze_scores(str(work), ["model"], language, "all")
        summary[language] = {
            name: value["model"]
            for name, value in zip(
                (
                    "semantics",
                    "quality",
                    "overall",
                    "intersection_semantics",
                    "intersection_quality",
                    "intersection_overall",
                ),
                values,
                strict=True,
            )
        }
    return finite(summary)


def rise(code: Path, bundle: dict, work: Path, workers: int) -> dict:
    import pandas as pd

    module = official_module(code, "gpt_eval")
    module.api_key = os.environ["OPENAI_API_KEY"]
    module.api_base = (
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        + "/chat/completions"
    )
    transport = JudgeAPI(work, bundle)
    original_requests = module.requests
    module.requests = SimpleNamespace(post=partial(transport.post, return_errors=True))
    cache, xlsx = work / "official.pkl", work / "official_judge.xlsx"
    # Official cache otherwise retains malformed API responses indefinitely.
    partial_judges = {}
    if cache.exists() and xlsx.exists():
        old = module.load(str(cache))
        scores = pd.read_excel(xlsx)
        failed = scores.loc[scores["match_log"] != "succeed", "index"]
        if len(failed):
            for key in failed:
                raw = old.pop(key, {})
                write_json(work / "responses" / f"{key}-{time.time_ns()}.json", raw)
                partial_judges[key] = {
                    name: value
                    for name, value in raw.items()
                    if value is None
                    or (isinstance(value, str) and module.extract(value))
                }
            module.dump(old, str(cache))
    sys.argv = [
        "gpt_eval.py",
        "--data",
        str(work / "data.json"),
        "--input",
        str(work / "sources"),
        "--output",
        str(work),
        "--model",
        "official",
        "--nproc",
        str(workers),
    ]
    original_generate = module.gpt_generate
    original_evaluate = module.eval_vanilla

    def evaluate(item, **kwargs):
        if item["index"] in partial_judges:
            item = item.copy()
            item["judge"] = partial_judges[item["index"]]
        return original_evaluate(item, **kwargs)

    module.eval_vanilla = evaluate
    if "judge" in bundle:
        module.gpt_generate = partial(original_generate, model=bundle["judge"]["model"])
    try:
        module.main()
    finally:
        module.gpt_generate = original_generate
        module.eval_vanilla = original_evaluate
        module.requests = original_requests
    rows = pd.read_excel(xlsx)
    complete = int((rows["match_log"] == "succeed").sum())
    write_json(
        work / "coverage.json",
        {"expected": len(bundle["records"]), "complete": complete},
    )
    if complete != len(bundle["records"]):
        raise IncompleteJudgements(
            "RISE contains unparseable judgements; "
            "official raw outputs retained, rerun --resume"
        )
    table = pd.read_csv(work / "official_judge.csv").set_index("-")
    return finite(
        {
            "accuracy": float(table.loc["Overall", "Accuracy"]),
            "official_table": table.to_dict(orient="index"),
        }
    )


def mse(code: Path, bundle: dict, work: Path, workers: int) -> dict:
    import pandas as pd
    from PIL import Image

    module = official_module(code, "compute_score")
    records = bundle["records"]

    class ResultRows:
        def __len__(self):
            return len(records)

        def __iter__(self):
            for row in records:
                yield {
                    f"turn_{i}_image": Image.open(path)
                    for i, path in enumerate(row["outputs"])
                }

    def local_dataset(name, **kwargs):
        if name == "leigangqu/MSE-Bench":
            return {
                "train": [
                    {**r["metadata"], "image": Image.open(r["source"])} for r in records
                ]
            }
        if name == "leigangqu/MSE-Bench-results":
            return ResultRows()
        raise ValueError(f"Unexpected upstream dataset: {name}")

    module.load_dataset = local_dataset
    res = work / "scores.json"
    # Preserve successful sessions; let the official loop retry unsuccessful API calls.
    invalid = False
    for cached in (work / "scores").glob("*.json"):
        if "error" in read_json(cached):
            cached.rename(cached.with_suffix(f".failed-{time.time_ns()}"))
            invalid = True
    if invalid and res.exists():
        res.rename(res.with_suffix(f".failed-{time.time_ns()}"))
    original = module.evaluate_with_gpt
    original_openai = module.openai
    original_tries = module.NUM_TRY
    transport = JudgeAPI(work, bundle)
    settings = bundle.get("judge", judge_settings("mse"))

    def client(**kwargs):
        value = transport.client(original_openai.OpenAI, **kwargs)
        return (
            mse_responses_via_chat(value, audit_dir=work / "transport")
            if settings["api"] == "chat"
            else value
        )

    # Bound native parse attempts (previously 20 x 3 x SDK retries). Exhausted
    # transport requests are memoized within this pass, not sent again.
    module.NUM_TRY = 3
    module.openai = SimpleNamespace(OpenAI=client)

    def capture(client_info, editing_prompt, base64_images):
        key = digest([client_info[1], editing_prompt, base64_images])
        cached = work / "turn-responses" / f"{key}.json"
        if cached.exists():
            return read_json(cached)["response"]
        result = original(client_info, editing_prompt, base64_images)
        write_json(
            work / "responses" / f"{digest(editing_prompt)}-{time.time_ns()}.json",
            {"prompts": editing_prompt, "response": result},
        )
        # Keep valid earlier turns if a later turn has a transient API failure.
        try:
            parsed = module.parse_mixed_string(result)
            valid = all(
                isinstance(parsed[k], (int, float)) and 0 <= parsed[k] <= 10
                for k in ("prompt_following", "consistency")
            )
        except (TypeError, ValueError, KeyError):
            valid = False
        if valid:
            write_json(cached, {"response": result})
        return result

    module.evaluate_with_gpt = capture
    sys.argv = [
        "compute_score.py",
        "--model_name",
        "model",
        "--res_path",
        str(res),
        "--image_cache_dir",
        str(work / "image_cache"),
        "--num_workers",
        str(workers),
        "--api_model",
        settings["model"],
    ]
    # Call main (not __main__): Unix fork preserves only the local I/O substitutions.
    import multiprocessing

    multiprocessing.set_start_method("fork", force=True)
    try:
        try:
            module.main()
        except AssertionError as error:
            # Native aggregation asserts if any session exhausted its retries.
            # Keep the session caches and route only that case to recovery.
            if not res.exists() or not any("error" in r for r in read_json(res)):
                raise
            raise IncompleteJudgements(
                "Incomplete MSE judgements; see scores/"
            ) from error
    finally:
        module.evaluate_with_gpt = original
        module.openai = original_openai
        module.NUM_TRY = original_tries
    results = read_json(res)
    if len(results) != len(records) or any(
        "error" in r or "turn1" not in r for r in results
    ):
        raise IncompleteJudgements("Incomplete MSE judgements")
    table = pd.read_csv(work / "tmp_data/csv/scores.json.csv")
    write_json(
        work / "coverage.json",
        {
            "expected": len(records),
            "complete": len(results),
            "note": "Official early-stop on first unsuccessful edit; "
            "denominator is all sessions.",
        },
    )
    metrics = {"turns": table.set_index("Turn").to_dict(orient="index")}
    jpeg_audits = list((work / "transport").glob("jpeg-*.json"))
    if jpeg_audits:
        # Existing report rendering preserves string notes without treating them
        # as numeric metrics. Successful pre-fallback scores remain untouched.
        metrics["note"] = (
            f"MSE transport variant mse-jpeg-fallback-v1: {len(jpeg_audits)} audited "
            "oversized request payload(s) used lossy JPEG quality=98, 4:4:4 after "
            "lossless WebP was insufficient. No resizing; cached images, judge, "
            "prompts and aggregation unchanged. This is mixed-encoding scoring, "
            "not strictly pixel-identical native scoring. "
            f"Audits: {work / 'transport'}."
        )
    return metrics


def _checked_edival_turn(evaluate, models, *args):
    """Do not let native VLM exception-to-'no' handling become a cached score."""
    original_chat = models.vlm_model.chat
    errors = []

    def checked_chat(*a, **kw):
        try:
            return original_chat(*a, **kw)
        except Exception as error:
            errors.append(error)
            raise

    models.vlm_model.chat = checked_chat
    try:
        result = evaluate(models, *args)
        if errors:
            raise RuntimeError(
                "EdiVal VLM request failed; refusing to count it as an editing "
                f"failure: {redact_secret(str(errors[0]))}"
            ) from errors[0]
        return result
    finally:
        models.vlm_model.chat = original_chat


def edival(code: Path, bundle: dict, work: Path, workers: int) -> dict:
    job = read_json(Path(bundle["job"]))
    root = Path(job["eval_root"])
    # Native repo identifiers resolve offline through our pinned cache refs.
    # No model implementation or from_pretrained method is patched.
    from .judge_assets import verify_edival_judges

    verify_edival_judges(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    module = official_module(code, "eval")
    # Native relative model paths are mapped into this scoring workspace.
    grounding = work / "GroundingDINO"
    grounding.mkdir(exist_ok=True)
    for name, target in {
        "groundingdino": code / "GroundingDINO/groundingdino",
        "weights": root / "models/groundingdino",
    }.items():
        link = grounding / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    original = module.evaluate_turn_multipass

    def cached_turn(models, paths, instruction_data, turn, index):
        path = work / "items" / f"{index}-{turn}.json"
        if path.exists():
            return read_json(path)
        result = _checked_edival_turn(
            original, models, paths, instruction_data, turn, index
        )
        write_json(path, module._make_json_serializable(result))
        return result

    module.evaluate_turn_multipass = cached_turn
    try:
        module.evaluate(
            str(work / "instructions.csv"),
            str(work),
            str(work / "scores"),
            3,
            modes=["multipass"],
        )
    finally:
        module.evaluate_turn_multipass = original
    rates = read_json(work / "scores/multipass/task_rate.json")
    per_image = read_json(work / "scores/multipass/image_rate.json")
    if len(per_image) != len(bundle["records"]) or any(
        len(v) != 3 or any(x is None for x in v) for v in per_image.values()
    ):
        raise RuntimeError("Incomplete EdiVal instruction judgements")
    write_json(
        work / "coverage.json",
        {
            "expected": len(bundle["records"]) * 3,
            "complete": sum(map(len, per_image.values())),
        },
    )

    def average(xs):
        return sum(xs) / len(xs) if xs else None

    return finite(
        {
            "instruction_following": average(rates["overall"]),
            "turns": {
                k: {name: average(xs) for name, xs in metrics.items()}
                for k, metrics in rates.items()
                if k != "overall"
            },
            "note": "Official multipass; optional HPSv3 is null "
            "if not installed (not a zero score).",
        }
    )


from .qwenimagebench import score as qwenimagebench
from .bizgeneval import score as bizgeneval

SCORERS = {
    "qwenimagebench": qwenimagebench,
    "bizgeneval": bizgeneval,
    "gedit": gedit,
    "rise": rise,
    "imgedit-basic": imgedit,
    "mse": mse,
    "edival": edival,
}
