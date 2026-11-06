import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tailsteer.evaluation import add_preserve_objects as judge
from tailsteer.evaluation.api_retry import InvalidJudgement, JudgeAPIError
from tailsteer.evaluation.common import file_digest, read_json, write_json


def response(objects, verdict="yes"):
    return {
        "model": "gpt-6.1-sol-2026-09-29",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "items": [
                                dict(
                                    id=o["id"],
                                    verdict=verdict,
                                    evidence="Visible test instance",
                                )
                                for o in objects
                            ]
                        }
                    )
                },
            }
        ],
    }


def fixture(tmp_path):
    from test_add_preserve import fixture_config

    config, _ = fixture_config(tmp_path)
    settings = config["evaluation"]["add_preserve"]
    settings["object_judge"] = {"enabled": True, "workers": 1, "scoring_passes": 2}
    manifest_path = Path(settings["manifest"])
    manifest = read_json(manifest_path)
    row = manifest["items"][0]
    row.update(size=[160, 128], bbox=[40, 32, 120, 96])
    manifest["source_decoration"] = {"width_pixels": 3}
    for r, name in zip(row["rounds"], ["a small red apple", "a small green apple"]):
        r["object"] = name
    write_json(manifest_path, manifest)
    settings["manifest_sha256"] = file_digest(manifest_path)
    output = Path(config["run"]["output_dir"])
    (output / "tasks/001/base").mkdir(parents=True)
    raw = (manifest_path.parent / row["image"]).read_bytes()
    (output / "tasks/001/source.png").write_bytes(raw)
    for t in (1, 2):
        (output / f"tasks/001/base/round-{t:02d}.png").write_bytes(raw)
    return config, output, row


def test_prompt_and_native_fixed_crop(tmp_path):
    config, output, row = fixture(tmp_path)
    path = output / "tasks/001/base/round-01.png"
    raw, box = judge.crop_image(path, row, 3)
    import io

    assert box == [43, 35, 117, 93]
    assert Image.open(io.BytesIO(raw)).size == (74, 58)
    assert judge.checklist(row, 2) == [
        {"id": "o01", "object": "a red apple"},
        {"id": "o02", "object": "a green apple"},
    ]
    Image.new("RGB", (100, 100)).save(path)
    with pytest.raises(ValueError, match="must not resize"):
        judge.crop_image(path, row, 3)


def test_invalid_response_is_not_a_negative_score():
    objs = [{"id": "o01"}, {"id": "o02"}]
    r = response(objs, "unsure")
    assert all(
        i["verdict"] == "unsure" for i in judge.parse_response(r, objs, r["model"])
    )
    with pytest.raises(JudgeAPIError, match="unexpected model"):
        judge.parse_response(r, objs, "different-model")
    r["choices"][0]["finish_reason"] = "length"
    with pytest.raises(InvalidJudgement):
        judge.parse_response(r, objs, r["model"])
    r = response([{"id": "o01"}, {"id": "o01"}])
    with pytest.raises(InvalidJudgement):
        judge.parse_response(r, objs, r["model"])


def test_macro_round_normalization_and_missing_values():
    rows = [{"id": "a", "round_count": 2}, {"id": "b", "round_count": 2}]
    records = {
        ("a", 1): dict(n=1, yes=1, unsure=0),
        ("a", 2): dict(n=2, yes=0, unsure=1),
        ("b", 1): dict(n=1, yes=0, unsure=0),
        ("b", 2): dict(n=2, yes=2, unsure=0),
    }
    s = judge.aggregate(records, rows, "base")
    assert s["cumulative_object_accuracy"] == 0.5
    assert s["cumulative_object_accuracy_std"] == 0
    assert [r["mean"] for r in s["rounds"]] == [0.5, 0.5]
    del records[("b", 2)]
    s = judge.aggregate(records, rows, "base")
    assert s["complete_images"] == 1 and s["scored"] == 3
    assert s["rounds"][1]["n"] == 1 and s["rounds"][1]["unsure"] == 1


def test_resume_retry_changed_image_and_blinded_payload(tmp_path, monkeypatch):
    config, output, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(judge, "configure_api", lambda *args: "https://example.test/v1")

    def post(self, url, **kwargs):
        payload = kwargs["json"]
        objects = json.loads(
            payload["messages"][1]["content"][0]["text"].split("\n", 1)[1]
        )
        calls.append(payload)
        assert '"base"' not in json.dumps(payload)
        assert "tailsteer" not in json.dumps(payload)
        assert payload["messages"][0]["content"] == judge.PROMPT
        # One malformed second-round response, then recover without repeating round 1.
        data = response(objects)
        if len(calls) == 2:
            data["choices"][0]["message"]["content"] = "{}"
        return SimpleNamespace(json=lambda: data)

    monkeypatch.setattr(judge.JudgeAPI, "post", post)
    assert judge.score(config) == 0 and len(calls) == 3
    summary = output / "add-preserve/object-judge/base/summary.json"
    assert read_json(summary)["complete"]
    assert judge.score(config) == 0 and len(calls) == 3
    Image.new("RGB", (160, 128), "red").save(output / "tasks/001/base/round-02.png")
    assert judge.score(config) == 0 and len(calls) == 4
    (output / "tasks/001/base/round-02.png").unlink()
    assert judge.score(config) == 1
    assert not read_json(summary)["complete"]
    assert not (summary.parent / "COMPLETE").exists()
    assert len(calls) == 4


def test_default_is_enabled_and_can_be_disabled():
    c = {"evaluation": {"add_preserve": {}}}
    assert judge.settings_for(c)["enabled"] is True
    c["evaluation"]["add_preserve"]["object_judge"] = {"enabled": False}
    assert judge.score(c) == 0


@pytest.mark.parametrize(
    "status,code,expected_calls",
    [
        (400, "content_policy_violation", [1, 2]),
        (401, "invalid_api_key", [1]),
        (400, "model_not_found", [1]),
    ],
)
def test_image_rejection_does_not_abort_other_samples(
    tmp_path, monkeypatch, status, code, expected_calls
):
    config, output, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(judge, "configure_api", lambda *args: "https://example.test/v1")

    def post(self, url, **kwargs):
        objects = json.loads(
            kwargs["json"]["messages"][1]["content"][0]["text"].split("\n", 1)[1]
        )
        calls.append(len(objects))
        if len(objects) == 1:
            raise JudgeAPIError(
                "Judge request refused",
                status=status,
                retryable=False,
                body=json.dumps({"error": {"code": code}}),
            )
        return SimpleNamespace(json=lambda: response(objects))

    monkeypatch.setattr(judge.JudgeAPI, "post", post)
    assert judge.score(config) == 1
    assert calls == expected_calls
    summary = read_json(output / "add-preserve/object-judge/base/summary.json")
    assert not summary["complete"]
    assert summary["scored"] == len(expected_calls) - 1
    assert (
        summary["rounds"][0]["n"] == 0
    )  # A rejection is never counted as zero credit.


def test_default_metric_entrypoint_runs_judge_and_propagates_failure(
    tmp_path, monkeypatch
):
    from tailsteer.evaluation import add_preserve as lp

    config, _, _ = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(
        lp, "OutsideMetrics", lambda _: lambda *args: dict.fromkeys(lp.METRICS, 1.0)
    )
    monkeypatch.setattr(judge, "score", lambda c: calls.append(c) or 1)
    assert lp.score(config) == 1  # Pixel success alone cannot hide an API failure.
    assert len(calls) == 1
    assert lp.score(config, preservation_only=True) == 0
    assert len(calls) == 1
