import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tailsteer.inference import sensenova_runner as runner
from tailsteer.inference.animation import render_sequences
from tailsteer.inference.backends.sensenova import SenseNovaInferenceBackend
from tailsteer.models.sensenova.model import SenseNovaModel


def test_sampler_receives_ordered_images_and_official_prompt():
    calls = []
    model = object.__new__(SenseNovaModel)

    def sample(tokenizer, prompt, images, **kwargs):
        calls.append((prompt, images, kwargs))
        return torch.zeros(1, 3, 64, 64)

    model.runtime = SimpleNamespace(
        model=SimpleNamespace(it2i_generate=sample), tokenizer=object()
    )
    source, first, second = (Image.new("RGB", (64, 64)) for _ in range(3))
    kwargs = dict(
        prompt="Edit image 1 using image 2 and image 3",
        seed=123,
        width=64,
        height=64,
        config={"num_steps": 2},
    )
    model.generate_image(source, condition_images=(first, second), **kwargs)
    assert calls[-1][0] == kwargs["prompt"]  # Official code adds Image-N tags.
    assert calls[-1][1] == [source, first, second]
    assert calls[-1][2]["seed"] == 123
    assert calls[-1][2]["image_size"] == (64, 64)
    model.generate_image(source, **kwargs)
    assert calls[-1][0] == "<image>\n" + kwargs["prompt"]
    assert calls[-1][1] == [source]
    with pytest.raises(ValueError, match="placeholders"):
        model.generate_pixels(source, **{**kwargs, "prompt": "<image><image>"})


@pytest.fixture
def multi_config(tmp_path):
    for name, value in (("source", 0), ("a", 100), ("b", 200)):
        Image.new("RGB", (512, 512), (value,) * 3).save(tmp_path / f"{name}.png")
    rows = [
        {
            "id": "example",
            "image": "source.png",
            "rounds": [
                {
                    "prompt": "First edit",
                    "seed": 9,
                    "steps": 2,
                    "condition_images": ["a.png", "b.png"],
                },
                {
                    "prompt": "Second edit",
                    "seed": 10,
                    "steps": 2,
                    "condition_images": ["b.png"],
                },
            ],
        }
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": rows}))
    return {
        "model": {},
        "seed": 42,
        "run": {"output_dir": str(tmp_path / "generation")},
        "schedule": runner.SENSENOVA_FIXED_SCHEDULE,
        "inference": {
            "tasks": str(path),
            "image_root": str(tmp_path),
            "export_path": str(tmp_path / "export"),
            "num_steps": 2,
            "feedback": {"mode": "image"},
            "multi_condition": {"max_static_references": 2},
            "policies": {
                name: {"segments": [{"start": 0, "end": 2, "actor": actor}]}
                for name, actor in (
                    ("base", "base"),
                    ("tailsteer", "refined"),
                )
            },
        },
    }


def test_multi_image_runner_independent_chains_resume_and_animation(
    multi_config, tmp_path, monkeypatch
):
    calls = []

    def generate(source, *, condition_images, prompt, seed, policy, config):
        current = source.getpixel((0, 0))[0]
        refs = tuple(image.getpixel((0, 0))[0] for image in condition_images)
        calls.append((policy.name, current, refs, prompt, seed))
        actor = policy.actor_at(0)
        increment = 1 if actor == "base" else 10
        return Image.new("RGB", (512, 512), (current + increment,) * 3), (actor,) * 2

    engine = SimpleNamespace(
        image_multiple=32,
        generate=generate,
        actors=SimpleNamespace(
            manifest={"step": 2000},
            export=SimpleNamespace(schema_version=2),
            parameter_count=3,
            generation_parameter_count=2,
            conditioning_parameter_count=1,
        ),
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(runner, "create_model", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "SenseNovaPolicyInference", lambda *_: engine)
    manifest = runner.run_sensenova_inference(multi_config)
    assert [row[:3] for row in calls] == [
        ("base", 0, (100, 200)),
        ("tailsteer", 0, (100, 200)),
        ("base", 1, (200,)),
        ("tailsteer", 10, (200,)),
    ]
    assert calls[0][3:] == calls[1][3:] and calls[2][3:] == calls[3][3:]
    records = manifest["tasks"][0]["results"]
    assert records[2]["condition_images"] == ["b.png"]
    assert records[2]["primary_source"] == "round_1.output"
    assert manifest["distributed"]["strategy"] == "whole_chain_balanced_by_input_count"
    assert runner.run_sensenova_inference(multi_config, resume=True) == manifest
    assert len(calls) == 4
    gallery = render_sequences(
        tmp_path / "generation", tmp_path / "animations", mp4=False
    )
    assert gallery.is_file()
    assert list((tmp_path / "animations").rglob("*.gif"))


@pytest.mark.parametrize(
    "case", ["unconfigured", "model_native", "missing_reference", "too_many"]
)
def test_reject_invalid_multi_image_before_model_loading(multi_config, case):
    i = multi_config["inference"]
    if case == "unconfigured":
        del i["multi_condition"]
        error, message = ValueError, "multi_condition"
    elif case == "model_native":
        i["feedback"] = "model_native"
        error, message = ValueError, "feedback=image"
    elif case == "missing_reference":
        from pathlib import Path

        (Path(i["image_root"]) / "a.png").unlink()
        error, message = FileNotFoundError, "a.png"
    else:
        i["multi_condition"]["max_static_references"] = 1
        error, message = ValueError, "static references"
    with pytest.raises(error, match=message):
        runner.run_sensenova_inference(multi_config)


def test_backend_requires_explicit_multi_image_feedback(multi_config, tmp_path):
    inference = multi_config["inference"]
    context = SimpleNamespace(
        training_config={"model": {}},
        suite={
            "tasks": inference["tasks"],
            "image_root": str(tmp_path),
            "num_steps": 2,
            "backends": {
                "sensenova": {
                    "inference": {
                        "multi_condition": {"max_static_references": 2},
                        "feedback": "image",
                    }
                }
            },
        },
        export_path=tmp_path / "export",
        policies=inference["policies"],
        inference_overrides={},
        execution=None,
        evaluation={},
        output_dir=tmp_path / "out",
    )
    backend = SenseNovaInferenceBackend()
    cfg = backend.build_config(context)
    assert cfg["inference"]["multi_condition"] == {"max_static_references": 2}
    assert cfg["inference"]["feedback"] == {"mode": "image"}
    context.inference_overrides = {"feedback": "model_native"}
    with pytest.raises(ValueError, match="feedback=image"):
        backend.build_config(context)
