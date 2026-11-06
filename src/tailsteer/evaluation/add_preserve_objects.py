"""Crop-only cumulative object judging; independent of torch and generation.

The cumulative object checklist and grading prompt are fixed. Missing API responses are not scores.
Receipts are keyed by image, crop, cumulative checklist and judge configuration.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from threading import Event

from .api import configure_api, redact_secret
from .api_retry import JudgeAPI, JudgeAPIError, InvalidJudgement
from .common import PROJECT, atomic_text, digest, file_digest, read_json, write_json
from .add_preserve import dataset_rows

PROMPT = (
    "This is a crop of the allowed editing area. For each requested object, "
    "check whether a recognizable instance with the specified color/material "
    "is visibly present and its body is fully inside the crop. Ignore size, "
    "shadows and background quality. Answer yes, no, or unsure, with brief "
    'visual evidence. Return only JSON: {"items":[{"id":"o01",'
    '"verdict":"yes|no|unsure","evidence":"brief evidence"}]}. '
    "Include every requested ID once."
)
PROTOCOL = {
    "version": "crop-objects-a-v1",
    "prompt": PROMPT,
    "crop": "fixed_manifest_bbox_inset_by_source_border_native_pixels",
    "checklist": "all_objects_requested_through_current_round_without_small",
    "verdicts": ["yes", "no", "unsure"],
    "unsure_credit": 0,
    "aggregation": "equal_image_mean_of_per_round_yes_divided_by_round",
    "std": "population_std_of_per_image_mean_over_rounds",
    "api_failure": "missing_not_zero",
}


def settings_for(config):
    from tailsteer.utils.config import load_config

    settings = load_config(PROJECT / "configs/evaluation/add_preserve_objects.yaml")
    settings.update(config["evaluation"]["add_preserve"].get("object_judge", {}))
    if not settings["enabled"]:
        return settings
    if settings["workers"] < 1 or settings["scoring_passes"] < 1:
        raise ValueError("Object judge workers/scoring_passes must be positive")
    return settings


def crop_image(path, row, border):
    """Never detect the generated border, resize, or alter a saved output image."""
    from PIL import Image

    x0, y0, x1, y1 = row["bbox"]
    box = (x0 + border, y0 + border, x1 - border, y1 - border)
    with Image.open(path) as image:
        w, h = image.size
        if list(image.size) != row["size"]:
            raise ValueError("Output dimensions changed; object judge must not resize")
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError(f"Invalid fixed interior crop: {box}")
        buffer = io.BytesIO()
        image.convert("RGB").crop(box).save(buffer, format="PNG")
    return buffer.getvalue(), list(box)


def checklist(row, turn):
    return [
        {"id": f"o{i:02d}", "object": r["object"].replace("a small ", "a ", 1)}
        for i, r in enumerate(row["rounds"][:turn], 1)
    ]


def parse_response(response, objects, expected_model):
    if expected_model and response.get("model") != expected_model:
        raise JudgeAPIError(
            "Object judge returned an unexpected model: " + str(response.get("model")),
            retryable=False,
        )
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Judge response truncated or did not finish normally")
        items = json.loads(choice["message"]["content"])["items"]
        ids = [i["id"] for i in items]
        if len(ids) != len(objects) or set(ids) != {o["id"] for o in objects}:
            raise ValueError("Object IDs missing, duplicated or unexpected")
        if any(
            i["verdict"] not in PROTOCOL["verdicts"]
            or not isinstance(i.get("evidence"), str)
            or not i["evidence"].strip()
            for i in items
        ):
            raise ValueError("Invalid verdict or missing evidence")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise InvalidJudgement(str(error)) from error
    return sorted(items, key=lambda item: item["id"])


def aggregate(records, rows, policy):
    rounds = []
    max_round = max(r["round_count"] for r in rows)
    for turn in range(1, max_round + 1):
        eligible = [r for r in rows if r["round_count"] >= turn]
        found = [
            records[(r["id"], turn)] for r in eligible if (r["id"], turn) in records
        ]
        values = [r["yes"] / r["n"] for r in found]
        rounds.append(
            dict(
                round=turn,
                n=len(found),
                expected=len(eligible),
                mean=mean(values) if values else None,
                std=pstdev(values) if values else None,
                unsure=sum(r["unsure"] for r in found),
                objects=sum(r["n"] for r in found),
            )
        )
    # Full per-image chains only; incomplete coverage remains explicit.
    image_means = [
        mean(records[(r["id"], t)]["yes"] / t for t in range(1, r["round_count"] + 1))
        for r in rows
        if all((r["id"], t) in records for t in range(1, r["round_count"] + 1))
    ]
    return dict(
        policy=policy,
        scored=len(records),
        expected=sum(r["round_count"] for r in rows),
        rounds=rounds,
        complete_images=len(image_means),
        expected_images=len(rows),
        cumulative_object_accuracy=mean(image_means) if image_means else None,
        cumulative_object_accuracy_std=pstdev(image_means) if image_means else None,
    )


def write_report(destination, summary):
    write_json(destination / "summary.json", summary)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(summary["rounds"][0]))
    writer.writeheader()
    writer.writerows(summary["rounds"])
    atomic_text(destination / "rounds.csv", buf.getvalue())
    value = summary["cumulative_object_accuracy"]
    title = "pending" if value is None else f"{value * 100:.2f}%"
    lines = [
        "# Add & Preserve cumulative object accuracy (COA)",
        "",
        f"Scored {summary['scored']}/{summary['expected']}; complete={summary['complete']}.",
        f"COA: {title}. Partial coverage is not a final benchmark score.",
        "",
        "Each round: yes / cumulative requested objects; unsure receives no credit.",
        "Macro-average across images and rounds. API failures are missing, never zero.",
        "",
        "| Round | Images | COA (%) | Population std (%) | Unsure / objects |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in summary["rounds"]:
        cells = ["—" if r[k] is None else f"{100 * r[k]:.2f}" for k in ("mean", "std")]
        lines.append(
            f"| {r['round']} | {r['n']}/{r['expected']} | {' | '.join(cells)} | "
            f"{r['unsure']}/{r['objects']} |"
        )
    atomic_text(destination / "report.md", "\n".join(lines) + "\n")


def score(config, *, policies=None):
    """Score existing outputs only; successful immutable receipts resume automatically."""
    settings = settings_for(config)
    if not settings["enabled"]:
        return 0
    local = config["evaluation"]["add_preserve"]
    root, rows = dataset_rows(config)
    manifest_path = Path(local["manifest"])
    manifest = read_json(manifest_path)
    if (
        file_digest(manifest_path) != local["manifest_sha256"]
        or file_digest(Path(config["inference"]["tasks"])) != local["tasks_sha256"]
    ):
        raise ValueError("Frozen add-preserve manifest/tasks changed")
    border = manifest.get("source_decoration", {}).get("width_pixels", 0)
    policies = list(policies or config["inference"]["policies"])
    if not policies or not set(policies) <= set(config["inference"]["policies"]):
        raise ValueError("Unknown object-judge inference policy")
    output = Path(config["run"]["output_dir"])
    for row in rows:
        if file_digest(root / row["image"]) != row["image_sha256"]:
            raise ValueError(f"Frozen source changed: {row['id']}")
        if (
            file_digest(output / "tasks" / row["id"] / "source.png")
            != row["image_sha256"]
        ):
            # Compare pixels too: model runners can re-save the same source PNG.
            from PIL import Image

            with (
                Image.open(root / row["image"]) as ref,
                Image.open(output / "tasks" / row["id"] / "source.png") as src,
            ):
                if (
                    ref.size != src.size
                    or ref.convert("RGB").tobytes() != src.convert("RGB").tobytes()
                ):
                    raise ValueError(
                        f"Model source differs from frozen input: {row['id']}"
                    )
        if not all("object" in r for r in row["rounds"]):
            raise ValueError("Frozen dataset requires per-round object labels")
    key_file = Path(settings["key_file"]) if settings.get("key_file") else None
    base = configure_api(key_file, settings.get("base_url"))
    protocol = {
        **PROTOCOL,
        "model": settings["model"],
        "expected_model": settings["expected_model"],
        "reasoning_effort": settings["reasoning_effort"],
        "max_completion_tokens": settings["max_completion_tokens"],
        "image_detail": "high",
        "base_url": base,
    }
    identity = digest(
        {
            "protocol": protocol,
            "manifest": local["manifest_sha256"],
            "tasks": local["tasks_sha256"],
        }
    )
    failed = False
    for policy in policies:
        destination = output / "add-preserve/object-judge" / policy
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = _score_policy(
                output,
                destination,
                rows,
                border,
                policy,
                settings,
                protocol,
                identity,
                base,
                root,
            )
            failed |= code != 0
    return int(failed)


def _score_policy(
    output,
    destination,
    rows,
    border,
    policy,
    settings,
    protocol,
    identity,
    base,
    root,
    *,
    image_path=None,
    crop=crop_image,
):
    (destination / "COMPLETE").unlink(missing_ok=True)
    fatal = Event()
    records, errors = {}, {}
    rejected = set()
    jobs = [(row, turn) for row in rows for turn in range(1, row["round_count"] + 1)]

    def one(api, row, turn):
        key = (row["id"], turn)
        path = (
            image_path(row, turn)
            if image_path
            else output / "tasks" / row["id"] / policy / f"round-{turn:02d}.png"
        )
        receipt = destination / "records" / f"{row['id']}-{turn:02d}.json"
        record = {
            "id": row["id"],
            "round": turn,
            "policy": policy,
            "identity": identity,
        }
        try:
            if path is None:
                raise FileNotFoundError(
                    f"Missing validated image receipt: {row['id']}/{turn}"
                )
            image_hash = file_digest(path)
            if receipt.is_file():
                saved = read_json(receipt)
                if (
                    saved.get("status") == "complete"
                    and saved.get("identity") == identity
                    and saved.get("image_sha256") == image_hash
                ):
                    return key, saved, None
            if fatal.is_set():
                return (
                    key,
                    None,
                    "Judge model/authentication failed; rerun after fixing",
                )
            raw, box = crop(path, row, border)
            objects = checklist(row, turn)
            payload = dict(
                model=settings["model"],
                reasoning_effort=settings["reasoning_effort"],
                max_completion_tokens=settings["max_completion_tokens"],
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Requested objects:\n" + json.dumps(objects),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(raw).decode(),
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
            )
            record.update(
                image_sha256=image_hash,
                crop_sha256=hashlib.sha256(raw).hexdigest(),
                crop_box=box,
                objects=objects,
                request_sha256=digest(payload),
            )
            response = api.post(
                base + "/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                timeout=settings["timeout"],
            ).json()
            record["response"] = response
            items = parse_response(response, objects, settings["expected_model"])
            # Do not associate a response with a changed output during concurrent generation.
            if file_digest(path) != image_hash:
                raise ValueError(
                    "Output image changed during judging; retry after generation"
                )
            record.update(
                status="complete",
                items=items,
                n=len(items),
                yes=sum(i["verdict"] == "yes" for i in items),
                unsure=sum(i["verdict"] == "unsure" for i in items),
            )
            write_json(receipt, record)
            return key, record, None
        except (OSError, ValueError, KeyError, TypeError, JudgeAPIError) as error:
            if isinstance(error, JudgeAPIError) and not error.retryable:
                try:
                    code = json.loads(error.body or "{}").get("error", {}).get("code")
                except (ValueError, AttributeError, TypeError):
                    code = None
                if error.status == 400 and code == "content_policy_violation":
                    # A refused image is missing, not a global authentication failure.
                    # Keep the same payload/judge; do not retry it within this run.
                    rejected.add(key)
                else:
                    fatal.set()
            message = redact_secret(str(error))
            record.update(status="failed", error=message)
            write_json(receipt, record)
            return key, None, message

    def publish():
        summary = aggregate(records, rows, policy)
        summary.update(
            protocol=protocol,
            identity=identity,
            errors=errors,
            complete=len(records) == len(jobs) and not errors,
            dataset=str(root),
            output_dir=str(output),
            source="fixed_manifest_crop",
        )
        write_report(destination, summary)
        print(
            f"add-preserve objects {policy}: {len(records)}/{len(jobs)} "
            f"complete={summary['complete']}",
            flush=True,
        )
        return summary

    for attempt in range(settings["scoring_passes"]):
        pending = [
            (r, t)
            for r, t in jobs
            if (r["id"], t) not in records and (r["id"], t) not in rejected
        ]
        if not pending or fatal.is_set():
            break
        api = JudgeAPI(destination, {})
        with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
            futures = [pool.submit(one, api, r, t) for r, t in pending]
            for i, future in enumerate(as_completed(futures), 1):
                key, record, error = future.result()
                label = f"{key[0]}/{key[1]}"
                if error:
                    errors[label] = error
                else:
                    records[key] = record
                    errors.pop(label, None)
                if i % 50 == 0:
                    publish()
        publish()
    summary = publish()
    if summary["complete"]:
        (destination / "COMPLETE").touch()
    return 0 if summary["complete"] else 1
