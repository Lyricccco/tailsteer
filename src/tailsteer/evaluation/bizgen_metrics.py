"""BizGenEval score equations and image transport."""

import base64
from io import BytesIO
from statistics import mean


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {
            "true": True,
            "yes": True,
            "1": True,
            "false": False,
            "no": False,
            "0": False,
        }.get(value.lower().strip())
    return None


def render_prompt(template, questions):
    return template.format(
        checklist="\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)),
        expected_count=len(questions),
        required_keys=", ".join(map(str, range(1, len(questions) + 1))),
    )


def item_scores(item, answers, alpha, subset=None):
    indices = item.get(subset, []) if subset else range(1, len(item["questions"]) + 1)
    if not indices:
        return dict(accuracy=0.0, error_score=0.0, errors=0, n_questions=0)
    errors = sum(answers.get(str(i), {}).get("result") is not True for i in indices)
    return dict(
        accuracy=(len(indices) - errors) / len(indices),
        error_score=max(0.0, 1 - alpha * errors),
        errors=errors,
        n_questions=len(indices),
    )


def aggregate(records, key):
    return {
        group: dict(
            count=len(items),
            accuracy=mean(r["accuracy"] for r in items),
            error_score=mean(r["error_score"] for r in items),
        )
        for group in sorted({r[key] for r in records})
        for items in [[r for r in records if r[key] == group]]
    }


def encode_image(path):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        im.save(buffer, format="JPEG", quality=90, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )
