from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tailsteer.evaluation.common import file_digest, read_json, write_json
from tailsteer.evaluation import add_preserve as lp


def fixture_config(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    image = np.full((128, 160, 3), 100, dtype=np.uint8)
    mask = np.full((128, 160), 255, dtype=np.uint8)
    mask[32:96, 40:120] = 0
    image[mask == 0] = 255
    Image.fromarray(image).save(root / "source.png")
    Image.fromarray(mask).save(root / "mask.png")
    rounds = [
        {"prompt": "Add a small apple inside the white box", "seed": k}
        for k in (42, 43)
    ]
    row = {
        "id": "001",
        "category": "test",
        "image": "source.png",
        "keep_mask": "mask.png",
        "image_sha256": file_digest(root / "source.png"),
        "keep_mask_sha256": file_digest(root / "mask.png"),
        "rounds": rounds,
    }
    write_json(root / "manifest.json", {"items": [row]})
    write_json(
        root / "tasks.json",
        {
            "tasks": [
                {
                    "id": "001",
                    "image": "source.png",
                    "chain_rounds": True,
                    "rounds": rounds,
                }
            ]
        },
    )
    torch_home = tmp_path / "torch"
    weight = torch_home / "hub/checkpoints/alexnet-owt-7be5be79.pth"
    weight.parent.mkdir(parents=True)
    weight.touch()
    config = {
        "evaluation": {},
        "inference": {
            "tasks": str(root / "tasks.json"),
            "image_root": str(root),
            "feedback": {"mode": "image"},
            "policies": {"base": {}},
        },
        "run": {"output_dir": str(tmp_path / "output")},
    }
    import sys

    settings = {
        "manifest": str(root / "manifest.json"),
        "python": sys.executable,
        "torch_home": str(torch_home),
        "object_judge": {"enabled": False},
    }
    lp.configure(config, settings)
    return config, image


def test_frozen_data_and_feedback(tmp_path):
    config, _ = fixture_config(tmp_path)
    settings = config["evaluation"]["add_preserve"]
    assert settings["protocol"] == lp.PROTOCOL
    config["inference"]["feedback"]["mode"] = "latent"
    with pytest.raises(ValueError, match="feedback=image"):
        lp.configure(config, settings)
    config["inference"]["feedback"]["mode"] = "image"
    path = Path(config["inference"]["tasks"])
    tasks = read_json(path)
    tasks["tasks"][0]["rounds"][0]["prompt"] = "Changed prompt"
    write_json(path, tasks)
    with pytest.raises(ValueError, match="frozen"):
        lp.configure(config, settings)


def test_population_summary_and_incomplete_rounds():
    rows = [{"id": i, "round_count": 2} for i in ("a", "b")]
    records = {
        (i, "base", 1): {"metrics": dict.fromkeys(lp.METRICS, v)}
        for i, v in (("a", 2.0), ("b", 4.0))
    }
    first, second = lp.aggregate(records, rows, ["base"])
    assert (first["n"], first["expected"], first["psnr_mean"], first["psnr_std"]) == (
        2,
        2,
        3,
        1,
    )
    assert second["n"] == 0 and second["lpips_mean"] is None


def test_resume_accepts_worker_config_provenance(tmp_path, monkeypatch):
    import yaml

    config, _ = fixture_config(tmp_path)
    output = Path(config["run"]["output_dir"])
    output.mkdir()
    (output / "config.yaml").write_text(
        yaml.safe_dump({**config, "_config_path": str(tmp_path / "plan.yaml")})
    )

    def launch(*args, **kwargs):
        raise RuntimeError("scorer launch reached")

    monkeypatch.setattr(lp.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="scorer launch reached"):
        lp.run_with_metrics(
            ["unused", "--resume"],
            cwd=tmp_path,
            environment={},
            config_path=tmp_path / "plan.yaml",
            config=config,
        )


def test_incremental_scores_resume_and_changed_png(tmp_path, monkeypatch):
    config, reference = fixture_config(tmp_path)
    calls = []

    def metric(ref, pred, keep):
        value = float(np.abs(ref.astype(float) - pred)[keep].mean())
        calls.append(value)
        return dict.fromkeys(lp.METRICS, value)

    monkeypatch.setattr(lp, "OutsideMetrics", lambda _: metric)
    task = Path(config["run"]["output_dir"]) / "tasks/001"
    (task / "base").mkdir(parents=True)
    Image.fromarray(reference).save(task / "source.png")
    first = task / "base/round-01.png"
    second = task / "base/round-02.png"
    Image.fromarray(reference).save(first)
    assert lp.score(config) == 1  # Missing round two is never marked complete.
    summary = task.parents[1] / "add-preserve/summary.json"
    assert read_json(summary)["scored"] == 1
    second.write_bytes(b"unfinished PNG")
    assert lp.score(config) == 1
    Image.fromarray(reference).save(second)
    assert lp.score(config) == 0
    assert len(calls) == 2
    assert lp.score(config) == 0 and len(calls) == 2  # Receipt reuse.
    modified = reference.copy()
    modified[:10, :10] = 0
    Image.fromarray(modified).save(second)
    assert lp.score(config) == 0 and len(calls) == 3
    assert read_json(summary)["rounds"][1]["psnr_mean"] > 0
    first.unlink()
    assert lp.score(config) == 1
    assert not (summary.parent / "COMPLETE").exists()


def test_real_masked_metrics_only_penalize_outside():
    pytest.importorskip("lpips")
    home = Path(
        __import__("os").environ.get("TORCH_HOME", "~/.cache/torch")
    ).expanduser()
    if not (home / "hub/checkpoints/alexnet-owt-7be5be79.pth").exists():
        pytest.skip("Offline AlexNet weights unavailable")
    metric = lp.OutsideMetrics(home)
    ref = np.random.default_rng(42).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    keep = np.ones((128, 160), dtype=bool)
    keep[32:96, 40:120] = False
    pred = ref.copy()
    pred[~keep] = 0
    assert metric(ref, pred, keep) == {"psnr": 120.0, "ssim": 1.0, "lpips": 0.0}
    pred[:32, :32] = 0
    changed = metric(ref, pred, keep)
    assert changed["psnr"] < 40 and changed["ssim"] < 1 and changed["lpips"] > 0
    with pytest.raises(ValueError, match="dimensions"):
        metric(ref, pred[:64], keep)


def test_launcher_rejects_changed_run_before_starting_scorer(tmp_path):
    config, _ = fixture_config(tmp_path)
    output = Path(config["run"]["output_dir"])
    output.mkdir()
    (output / "config.yaml").write_text("unrelated: config\n")
    args = dict(
        cwd=tmp_path, environment={}, config_path=tmp_path / "plan.yaml", config=config
    )
    with pytest.raises(FileExistsError, match="--resume"):
        lp.run_with_metrics(["unused"], **args)
    with pytest.raises(ValueError, match="resume config differs"):
        lp.run_with_metrics(["unused", "--resume"], **args)
    assert not (output / "add-preserve").exists()
