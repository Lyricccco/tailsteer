"""Native Hub shards preserve the existing inference actor contract."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tailsteer.models.checkpoint import checkpoint_manifest, open_weights
from tailsteer.models.native_transformer.inference import TransformerActors


@pytest.fixture
def native_repo(tmp_path):
    def build(model="qwen_image_21"):
        root = tmp_path / model
        folder = root if model == "sensenova_u1_5_8b_mot" else root / "transformer"
        folder.mkdir(parents=True)
        save_file(
            {"weight": torch.full((2, 2), 3, dtype=torch.bfloat16)},
            folder / "a.safetensors",
        )
        save_file(
            {"bias": torch.full((2,), 5, dtype=torch.bfloat16)},
            folder / "b.safetensors",
        )
        stem = (
            "model" if model == "sensenova_u1_5_8b_mot" else "diffusion_pytorch_model"
        )
        (folder / f"{stem}.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"weight": "a.safetensors", "bias": "b.safetensors"}}
            )
        )
        (root / "tailrefine_release.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "step": 2000,
                    "trainable_policy": {
                        "qwen_image_21": "denoiser_full",
                        "flux2_klein_9b": "full",
                        "sensenova_u1_5_8b_mot": "full_model",
                    }[model],
                }
            )
        )
        return root

    return build


@pytest.mark.parametrize(
    "model", ["flux2_klein_9b", "qwen_image_21", "sensenova_u1_5_8b_mot"]
)
def test_native_shards_keep_absolute_weights(native_repo, model):
    root = native_repo(model)
    manifest = checkpoint_manifest(root)
    prefix = "model." if model.startswith("sensenova") else ""
    with open_weights(
        root / manifest["weights"], prefix=manifest["weight_prefix"]
    ) as reader:
        assert set(reader.keys()) == {prefix + "weight", prefix + "bias"}
        assert torch.equal(
            reader.get_tensor(prefix + "weight"),
            torch.full((2, 2), 3, dtype=torch.bfloat16),
        )
        assert torch.equal(
            reader.get_tensor(prefix + "bias"),
            torch.full((2,), 5, dtype=torch.bfloat16),
        )
    if prefix:
        from tailsteer.models.sensenova.export import SenseNovaExportSpec

        spec = SenseNovaExportSpec.load(root)
        assert spec.trainable["conditioning_gradients"]
        assert spec.schema_version == 3 and spec.has_reference_actor


def test_native_actor_switch_preserves_base(native_repo):
    root = native_repo()
    layer = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    original = {k: v.detach().clone() for k, v in layer.named_parameters()}
    actors = TransformerActors(
        layer,
        root,
        model_name="qwen_image_21",
        export_format="tailsteer_qwen_image_21_transformer",
        offload_weights=True,
    )
    actors._activate("refined")
    assert torch.equal(
        layer(torch.ones((1, 2), dtype=torch.bfloat16)),
        torch.full((1, 2), 11, dtype=torch.bfloat16),
    )
    actors._activate("base")
    assert all(torch.equal(v, original[k]) for k, v in layer.named_parameters())


def test_hub_checkpoint_cli_keeps_both_policies(native_repo, tmp_path, monkeypatch):
    import tailsteer.cli as cli
    from tailsteer.inference.experiment import prepare_inference

    root = native_repo("flux2_klein_9b")
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **kw: calls.append((a, kw)) or str(root),
    )
    monkeypatch.setattr(cli, "launch", lambda *a, **kw: 0)
    output = tmp_path / "output"
    assert (
        cli.main(
            [
                "infer",
                "--checkpoint",
                "example/native",
                "--checkpoint-revision",
                "revision-a",
                "--output-dir",
                str(output),
                "--inference-suite",
                "cafe-20turn",
                "--dry-run",
            ]
        )
        == 0
    )
    assert calls[0][1]["revision"] == "revision-a"
    plan = prepare_inference(
        tmp_path / ".plans/output/model", step=2000, suite="cafe-20turn"
    )
    assert list(plan.config["inference"]["policies"]) == ["base", "tailsteer"]
    assert plan.export_path == root


def test_tailsteer_only_uses_complete_repo_without_base_download(
    native_repo, tmp_path, monkeypatch
):
    import tailsteer.cli as cli
    import yaml

    root = native_repo("flux2_klein_9b")
    monkeypatch.setattr(cli, "launch", lambda *a, **kw: 0)
    assert (
        cli.main(
            [
                "infer",
                "--checkpoint",
                str(root),
                "--policies",
                "tailsteer",
                "--output-dir",
                str(tmp_path / "output"),
                "--inference-suite",
                "cafe-20turn",
                "--dry-run",
            ]
        )
        == 0
    )
    config = yaml.safe_load((tmp_path / ".plans/output/model/config.yaml").read_text())
    assert config["model"]["path"] == str(root)
    assert config["inference_checkpoint"] == str(root)
