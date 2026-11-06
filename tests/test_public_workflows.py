"""Public commands preserve paper settings and standalone demo contracts."""

from pathlib import Path
import json
import pytest
from tailsteer.configuration import BACKBONES
from tailsteer.utils.config import load_config
from tailsteer.inference.experiment import prepare_inference, resolve_suite
from tailsteer.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "backbone,onset,last,lr,end_weight",
    [
        ("flux1-kontext-dev", 25, 29, 1e-5, 1),
        ("flux2-klein-base-9b", 46, 49, 1e-5, 1),
        ("qwen-image-2.1", 30, 39, 1e-6, 0.1),
        ("sensenova-u1.5-8b-mot", 42, 50, 1e-5, 1),
    ],
)
def test_paper_defaults(backbone, onset, last, lr, end_weight):
    c = load_config(ROOT / "configs/train" / f"{backbone}.yaml")
    t = c["training"]
    assert t["transition"] == dict(
        start_step=onset,
        supervision_step=last,
        alpha_schedule=dict(name="odds", odds_ratio=1.0),
    )
    assert t["learning_rate"] == lr
    assert t["teacher_guided_target"].get("endpoint_loss_weight", 1) == end_weight
    assert (t["max_steps"], t["warmup_steps"], t["lr_schedule_steps"]) == (
        2000,
        100,
        10000,
    )
    assert t["target_step_sampling"] == dict(
        strategy="sigma_beta", uniform_mix=0.8, beta=3.0
    )
    assert t["batch_size"] * t["gradient_accumulation_steps"] * 8 == 16
    assert (
        t["algorithm"] == "trajectory_preservation"
        and t["state_source"]["strategy"] == "cached_segment"
    )
    assert c["data"]["caches"][0]["mask_variant"] == "dilate5"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_base_only_plan_needs_no_checkpoint(backbone, tmp_path):
    import yaml

    c = load_config(ROOT / "configs/train" / f"{backbone}.yaml")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(c))
    p = prepare_inference(tmp_path, step=2000, suite="cafe-20turn", policies=["base"])
    assert list(p.config["inference"]["policies"]) == ["base"]
    assert "export_path" not in p.config["inference"]
    assert p.config["inference"]["feedback"]["mode"] == "image"
    assert p.config["inference"]["num_steps"] == {
        "flux1-kontext-dev": 30,
        "qwen-image-2.1": 40,
    }.get(backbone, 50)


def test_demo_assets_are_portable():
    for directory in (ROOT / "demo-testset").iterdir():
        data = json.loads((directory / "tasks.json").read_text())
        for task in data["tasks"]:
            paths = [task["image"]]
            for r in task["rounds"]:
                paths.extend(r.get("condition_images", []))
            for value in paths:
                if isinstance(value, dict):
                    value = value["image"]
                assert not Path(value).is_absolute()
                assert (directory / value).is_file()


def test_removed_suite_name_is_not_an_alias():
    with pytest.raises(FileNotFoundError):
        resolve_suite("local-preserve")


def test_source_manifest_grouping(tmp_path, monkeypatch):
    import tailsteer.cli as cli

    rows = [
        dict(
            id="a",
            source_id="shared",
            image="img.png",
            prompt="edit",
            task="local",
            seed=3,
        ),
        dict(
            id="b",
            source_id="shared",
            image="img.png",
            prompt="unchanged",
            task="identity",
            seed=4,
        ),
    ]
    source = tmp_path / "sources.jsonl"
    source.write_text("\n".join(map(json.dumps, rows)))
    monkeypatch.setattr(cli, "launch", lambda *a, **k: 0)
    assert (
        main(
            [
                "cache",
                "--sources",
                str(source),
                "--output-dir",
                str(tmp_path / "cache"),
                "--dry-run",
            ]
        )
        == 0
    )
    saved = [
        json.loads(s)
        for s in (tmp_path / ".plans/cache/sources.jsonl").read_text().splitlines()
    ]
    assert [s["source_identity"] for s in saved] == ["shared", "shared"]
    assert [s["task"] for s in saved] == ["local_edit", "noop"]


@pytest.mark.parametrize(
    "backbone", ["flux2-klein-base-9b", "qwen-image-2.1", "sensenova-u1.5-8b-mot"]
)
@pytest.mark.parametrize(
    "suite", ["lace-dress-dance", "yard-object-composition", "people-composition"]
)
def test_multi_reference_demos_keep_all_references(backbone, suite, tmp_path):
    import yaml
    from tailsteer.inference.multi_image_multiround import load_multi_image_tasks
    from tailsteer.inference.multiround import load_tasks

    config = load_config(ROOT / "configs/train" / f"{backbone}.yaml")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    plan = prepare_inference(tmp_path, step=2000, suite=suite, policies=["base"])
    inference = plan.config["inference"]
    tasks = load_multi_image_tasks(
        inference["tasks"],
        default_steps=inference["num_steps"],
        default_seed=42,
        max_static_references=inference["multi_condition"]["max_static_references"],
    )
    expected = json.loads(Path(inference["tasks"]).read_text())["tasks"]
    assert [[list(r.condition_images) for r in task.rounds] for task in tasks] == [
        [r["condition_images"] for r in task["rounds"]] for task in expected
    ]
    with pytest.raises(ValueError, match="multi_condition"):
        load_tasks(
            inference["tasks"], default_steps=inference["num_steps"], default_seed=42
        )
