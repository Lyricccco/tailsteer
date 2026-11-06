"""Single or paired policy summaries; no invented cross-benchmark combined score."""

from pathlib import Path

from .common import atomic_text, read_json, write_json
from .experiment import collect_outputs, score_directory


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}".strip("/")
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, (float, int)) and not isinstance(item, bool):
            result[name] = item
    return result


def write_report(output: Path) -> Path:
    job = read_json(output / "job.json")
    summaries = {
        p.parent.name: read_json(p)
        for p in sorted((output / "scores").glob("*/summary.json"))
    }
    if job.get("selected_policies") is not None:
        summaries = {
            policy: read_json(path)
            for policy in job["selected_policies"]
            if (path := score_directory(job, policy) / "summary.json").is_file()
        }
    if not summaries:
        raise ValueError(f"No scored results below {output}")
    if "generation" in job:
        for policy in collect_outputs(job):
            summaries.setdefault(
                policy, {"status": "not_scored", "error": "No score yet"}
            )
    complete = {
        k: _flatten(v["metrics"])
        for k, v in summaries.items()
        if v["status"] == "complete"
    }
    baseline = complete.get("base", {})
    judge = job.get("judge")
    notes = sorted(
        {
            result["metrics"]["note"]
            for result in summaries.values()
            if result.get("metrics", {}).get("note")
        }
    )
    if "baseline" in job:
        profile = job["baseline"]
        revision = profile.get("source", {}).get("revision")
        provenance = (
            f"revision `{revision}`"
            if revision
            else f"asset fingerprint `{profile['fingerprint']}`"
        )
        subject = f"Baseline: `{profile['name']}` · {provenance}"
    else:
        subject = f"Run: `{job['run_dir']}` · step {job['step']}"
    lines = [
        f"# {job['benchmark']}",
        "",
        subject,
        "",
        f"Sessions: {len(job['records'])}; subset: {job['subset']}; "
        f"feedback: {job['protocol']['feedback']}",
        "",
        (
            (
                "Differences are each policy − pretrained, not relative percentages. "
                if "baseline" in job
                else "Differences are refined − pretrained, not relative percentages. "
            )
            + "Metric direction depends on the benchmark."
            if baseline
            else "Absolute scores only; no pretrained comparison."
        ),
        "",
        *[f"Scorer note: {note}\n" for note in notes],
        *(
            [
                f"Judge variant: `{judge['model']}`; API: `{judge['api']}`. "
                "Official prompts/aggregation retained; "
                "not the unmodified native protocol.",
                "",
            ]
            if judge
            else []
        ),
        *(
            [f"Parser variant: `{judge['parser']}`; raw responses retained.", ""]
            if judge and "parser" in judge
            else []
        ),
        "| Policy | Metric | Value |" + (" Δ pretrained |" if baseline else ""),
        "|---|---|---:|" + ("---:|" if baseline else ""),
    ]
    for policy, metrics in complete.items():
        for name, value in metrics.items():
            delta = f"{value - baseline[name]:+.6f}" if name in baseline else "—"
            lines.append(
                f"| {policy} | {name} | {value:.6f} |"
                + (f" {delta} |" if baseline else "")
            )
    for policy, result in summaries.items():
        if result["status"] != "complete":
            lines += ["", f"Incomplete: {policy}: {result.get('error', '')}"]
    write_json(
        output / "report.json", {"job": str(output / "job.json"), "policies": summaries}
    )
    path = output / "report.md"
    atomic_text(path, "\n".join(lines) + "\n")
    return path
