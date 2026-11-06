"""Metric arithmetic and public benchmark-profile dispatch."""

from types import SimpleNamespace
import pytest
from tailsteer.evaluation.bizgen_metrics import item_scores, aggregate, render_prompt
from tailsteer.evaluation.bizgeneval import parse_answers
from tailsteer.evaluation.api_retry import InvalidJudgement


def test_bizgen_official_arithmetic():
    item = dict(questions=["a"] * 20, easy_qidxs=[1, 2, 3], hard_qidxs=[])
    answers = {str(i): {"result": i > 3} for i in range(1, 21)}
    assert item_scores(item, answers, 0.1) == dict(
        accuracy=0.85, error_score=0.7, errors=3, n_questions=20
    )
    assert item_scores(item, answers, 0.2, "easy_qidxs")["accuracy"] == 0
    assert item_scores(item, answers, 0.2, "hard_qidxs") == dict(
        accuracy=0.0, error_score=0.0, errors=0, n_questions=0
    )
    assert aggregate(
        [
            dict(domain="a", accuracy=0.5, error_score=0.8),
            dict(domain="a", accuracy=1, error_score=1),
        ],
        "domain",
    )["a"] == dict(count=2, accuracy=0.75, error_score=0.9)
    assert (
        render_prompt("{checklist}; {required_keys}; {expected_count}", ["a", "b"])
        == "1. a\n2. b; 1, 2; 2"
    )


def test_bizgen_malformed_json_remains_missing():
    def bad(raw):
        raise SyntaxError("bad response")

    upstream = SimpleNamespace(
        _parse_json_safe=bad, _extract_results_only=lambda *args: None
    )
    with pytest.raises(InvalidJudgement):
        parse_answers("bad", ["a"], upstream)


def test_evaluation_profile_continues_after_failure(tmp_path, monkeypatch):
    from tailsteer import cli
    from tailsteer.utils.config import load_config
    import yaml

    config = load_config(cli.ROOT / "configs/train/flux2-klein-base-9b.yaml")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    calls = []

    def launch(script, args, **kwargs):
        calls.append((script, args))
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(cli, "launch", launch)
    result = cli.main(
        [
            "infer",
            "--run-dir",
            str(tmp_path),
            "--policies",
            "base",
            "--inference-suite",
            str(cli.ROOT / "configs/evaluation/suites/t2i.yaml"),
            "--inference-suite",
            "cafe-20turn",
            "--dry-run",
        ]
    )
    assert result == 1 and len(calls) == 2
    assert calls[0][0] == "evaluate_sequence.py"
    assert "--continue-on-error" in calls[0][1]
    assert "--suite" in calls[0][1] and "--bench" not in calls[0][1]
    assert calls[1][0] == "inference_experiment.py"
