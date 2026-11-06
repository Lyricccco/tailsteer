"""Publication preserves source identities, frozen pixels and prompt contracts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "export_dataset", Path(__file__).resolve().parents[1] / "scripts/export_dataset.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_source_export_retains_grouping_and_instruction_order(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (32, 48), "blue").save(source)
    rows = [
        dict(
            id="image:p01",
            source_identity="dataset:image",
            source=str(source),
            prompt="Make the cup green.",
            task="local_edit",
            seed=7,
        ),
        dict(
            id="image:noop",
            source_identity="dataset:image",
            source=str(source),
            prompt="Keep the image exactly unchanged.",
            task="noop",
            seed=7,
        ),
    ]
    manifest, provenance = tmp_path / "source.jsonl", tmp_path / "provenance.jsonl"
    release.write_rows(manifest, rows)
    release.write_rows(provenance, [dict(source_id="dataset:image", category="photo")])
    output = tmp_path / "release"
    output.mkdir()
    summary = release.export_source(
        SimpleNamespace(
            input=manifest,
            provenance=provenance,
            output=output,
            image_root=tmp_path,
            workers=1,
        )
    )
    saved = release.read_rows(output / "sources.jsonl")
    assert summary["images"] == 1 and summary["trajectory_requests"] == 2
    assert [r["id"] for r in saved] == [r["id"] for r in rows]
    assert [r["prompt"] for r in saved] == [r["prompt"] for r in rows]
    assert [r["task"] for r in saved] == ["local", "identity"]
    assert saved[0]["source_id"] == saved[1]["source_id"] == "dataset:image"
    assert saved[0]["image"] == saved[1]["image"]
    assert release.sha256(output / saved[0]["image"]) == release.sha256(source)
    assert str(tmp_path) not in (output / "sources.jsonl").read_text()


def test_frozen_benchmark_rejects_changed_pixels(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "red").save(image)
    rounds = [
        dict(prompt=f"Add small object {i}.", object=f"object {i}", seed=i)
        for i in range(10)
    ]
    item = dict(
        id="01",
        category="photo",
        size=[32, 32],
        bbox=[8, 8, 24, 24],
        image="input.png",
        image_sha256="wrong",
        rounds=rounds,
    )
    release.write_json(tmp_path / "manifest.json", {"items": [item]})
    release.write_json(
        tmp_path / "tasks.json",
        {
            "tasks": [
                dict(
                    id="01",
                    image="input.png",
                    rounds=[{"prompt": r["prompt"], "seed": r["seed"]} for r in rounds],
                )
            ]
        },
    )
    release.write_rows(
        tmp_path / "provenance.jsonl", [dict(id="01", source_id="dataset:image")]
    )
    with pytest.raises(ValueError, match="Frozen asset changed"):
        release.export_benchmark(
            SimpleNamespace(
                input=tmp_path,
                provenance=tmp_path / "provenance.jsonl",
                output=tmp_path / "release",
                workers=1,
            )
        )


def test_asset_paths_cannot_escape_release(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        release.asset(tmp_path, "../outside.png")


@pytest.mark.parametrize(
    "dataset,repo,directory",
    [
        ("training-source", "Lyricccco/TailSteer-Source", "training"),
        ("add-preserve", "Lyricccco/Add-and-Preserve", "add-preserve"),
    ],
)
def test_default_dataset_download_is_pinned(
    dataset, repo, directory, tmp_path, monkeypatch
):
    from tailsteer.cli import DATASETS, main

    calls = []
    monkeypatch.setenv("TAILSTEER_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: calls.append((a, k))
    )
    assert main(["prepare", dataset]) == 0
    assert calls == [
        (
            (repo,),
            dict(
                repo_type="dataset",
                revision=DATASETS[dataset][1],
                local_dir=tmp_path / directory,
                max_workers=32,
            ),
        )
    ]
    calls.clear()
    assert main(["prepare", dataset, "--dataset-repo", "example/custom"]) == 0
    assert calls[0][1]["revision"] == "main"
    calls.clear()
    assert main(["prepare", dataset, "--revision", "custom-revision"]) == 0
    assert calls[0][1]["revision"] == "custom-revision"
