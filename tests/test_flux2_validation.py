from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from PIL import Image

from tailsteer.cache.dataset import cache_record, split_dataset
from tailsteer.cache.sample import CacheSample
from tailsteer.models.flux2.model import Flux2Model
from tailsteer.training.checkpoint import export_inference_model
from tailsteer.training.distributed import DistributedContext
from tailsteer.training.trainer import Trainer
from tailsteer.training.visual_validation import select_fixed_local_edit_indices


def test_flux2_fp32_master_is_opt_in_and_promotes_student_storage() -> None:
    raw = torch.nn.Sequential(
        torch.nn.Linear(3, 2),
        torch.nn.LayerNorm(2),
    ).to(dtype=torch.bfloat16)
    backend = Flux2Model.__new__(Flux2Model)
    backend.refined_transformer = raw
    backend.trainable_policy = "full"
    backend.trainable_config = {"policy": "full", "master_dtype": "model"}
    backend._fsdp_master_parameters_prepared = False

    # The omitted option retains the historical BF16 parameter path.
    backend.configure_trainable({"policy": "full"})
    assert all(parameter.dtype == torch.bfloat16 for parameter in raw.parameters())

    backend.configure_trainable({"policy": "full", "master_dtype": "float32"})
    backend.prepare_optimizer_parameters()
    assert all(parameter.dtype == torch.float32 for parameter in raw.parameters())
    assert backend.trainable_config["master_dtype"] == "float32"


def test_flux2_exposes_independent_teacher_for_fsdp_sharding() -> None:
    class TinyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.single_transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2)]
            )

    refined = TinyTransformer()
    teacher = TinyTransformer()
    backend = Flux2Model.__new__(Flux2Model)
    backend.refined_transformer = refined
    backend.pretrained_transformer = teacher

    groups = backend.auxiliary_sharding_groups()
    assert len(groups) == 1
    name, root, units = groups[0]
    assert name == "pretrained_teacher"
    assert root is teacher
    assert units == (
        *tuple(teacher.transformer_blocks),
        *tuple(teacher.single_transformer_blocks),
    )

    replacement = TinyTransformer()
    backend.replace_auxiliary_sharded_module(name, replacement)
    assert backend.pretrained_transformer is replacement

    backend.pretrained_transformer = backend.refined_transformer
    assert backend.auxiliary_sharding_groups() == ()


def test_flux2_checkpoint_export_is_directly_loadable_safetensors(tmp_path) -> None:
    root = torch.nn.Linear(3, 2)
    expected = {
        name: value.detach().clone() for name, value in root.state_dict().items()
    }
    exporting = Flux2Model.__new__(Flux2Model)
    exporting.refined_transformer = root
    exporting.trainable_policy = "full"
    context = DistributedContext("single", 0, 0, 1, torch.device("cpu"))

    destination = export_inference_model(
        tmp_path,
        name="step-00001000",
        model=exporting,
        root=root,
        step=1000,
        context=context,
    )

    assert (destination / "transformer.safetensors").is_file()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["step"] == 1000
    assert manifest["trainable_policy"] == "full"
    restored_root = torch.nn.Linear(3, 2)
    restored = Flux2Model.__new__(Flux2Model)
    restored.refined_transformer = restored_root
    restored.load_export(destination)
    for name, value in restored_root.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_flux2_visual_validation_runs_complete_tailsteer_schedule() -> None:
    model = Flux2Model.__new__(Flux2Model)
    model.pipe = SimpleNamespace(vae=object())
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.cfg_scale = 3.0
    calls = []

    def decode(latent, *, height, width):
        value = int((latent.float().mean().clamp(-1, 1).item() + 1.0) * 127.5)
        return Image.new("RGB", (width, height), (value, value, value))

    model.decode_latent = decode
    model.prepare_conditioning = lambda **_kwargs: object()
    model.initialize_latent = lambda *_args, **_kwargs: torch.zeros((1, 1, 2, 2))

    def predict(latent, point, _conditioning, *, actor):
        calls.append((point.step_index, actor, model.cfg_scale))
        return torch.ones_like(latent)

    model.predict_velocity = predict
    sample = CacheSample(
        latent=torch.zeros((1, 2, 2)),
        source_latent=torch.zeros((1, 2, 2)),
        keep_mask=torch.ones((1, 2, 2)),
        metadata={
            "sample_id": "held-out:49",
            "source_id": "held-out",
            "source_identity": "image:held-out",
            "prompt": "edit the held-out image",
            "seed": 7,
            "step_index": 49,
            "sigma": 0.015,
            "timestep": 15.0,
            "height": 32,
            "width": 32,
            "task": "local_edit",
        },
    )

    source, generated = model.generate_validation_sample(
        sample,
        {
            "num_steps": 3,
            "denoising_strength": 1.0,
            "noise_mode": "auto",
            "cfg_scale": 4.0,
        },
    )

    assert source.size == generated.size == (32, 32)
    assert source.getpixel((0, 0)) != generated.getpixel((0, 0))
    assert calls == [(0, "refined", 4.0), (1, "refined", 4.0), (2, "refined", 4.0)]
    assert model.cfg_scale == 3.0


class _RecordDataset:
    def __init__(self) -> None:
        self.records = tuple(
            {
                "sample_id": f"sample-{index}",
                "source_identity": f"source-{index}",
                "task": "local_edit",
            }
            for index in range(200)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return self.records[index]


def test_fixed_visual_sources_are_outside_training_split() -> None:
    train, validation = split_dataset(
        _RecordDataset(), seed=42, validation_fraction=0.2
    )
    selected = select_fixed_local_edit_indices(validation, count=10, seed=42)
    train_sources = {
        cache_record(train, index)["source_identity"] for index in range(len(train))
    }
    visual_sources = {
        cache_record(validation, index)["source_identity"] for index in selected
    }

    assert len(visual_sources) == 10
    assert train_sources.isdisjoint(visual_sources)


class _ValidationDataset:
    def __init__(self) -> None:
        self.samples = tuple(
            CacheSample(
                latent=torch.zeros((1, 2, 2)),
                source_latent=torch.zeros((1, 2, 2)),
                keep_mask=torch.ones((1, 2, 2)),
                metadata={
                    "sample_id": f"visual-{index}",
                    "source_id": f"visual-{index}",
                    "source_identity": f"image:visual-{index}",
                    "prompt": "edit" if index < 10 else "keep unchanged",
                    "seed": index,
                    "step_index": 49,
                    "sigma": 0.015,
                    "timestep": 15.0,
                    "height": 16,
                    "width": 16,
                    "task": "local_edit" if index < 10 else "noop",
                },
            )
            for index in range(12)
        )
        self.records = tuple(dict(sample.metadata) for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CacheSample:
        return self.samples[index]


class _ValidationModel:
    def __init__(self) -> None:
        self.root = torch.nn.Linear(1, 1)

    def get_trainable_module(self):
        return self.root

    def trainable_parameters(self):
        return tuple(self.root.parameters())

    @staticmethod
    def validation_inference_config(config):
        return {"sampler": "test", "num_steps": int(config["num_steps"])}

    @staticmethod
    def generate_validation_sample(sample, _config):
        source = Image.new("RGB", (16, 16), "black")
        generated = Image.new("RGB", (16, 16), "white")
        return source, generated


class _ValidationAlgorithm:
    @staticmethod
    def compute_loss(batch):
        losses = torch.tensor(
            [1.0 if item["task"] == "local_edit" else 2.0 for item in batch["metadata"]]
        )
        return SimpleNamespace(loss=losses.mean(), per_sample=losses)


def test_validation_writes_requested_loss_metrics_and_visual_outputs(tmp_path) -> None:
    model = _ValidationModel()
    dataset = _ValidationDataset()
    context = DistributedContext("single", 0, 0, 1, torch.device("cpu"))
    trainer = Trainer(
        model=model,
        algorithm=_ValidationAlgorithm(),
        optimizer=torch.optim.SGD(model.root.parameters(), lr=0.1),
        scheduler=None,
        train_dataset=dataset,
        validation_dataset=dataset,
        config={
            "seed": 42,
            "training": {
                "batch_size": 1,
                "num_workers": 0,
                "validation_batches": 0,
                "visual_validation": {
                    "enabled": True,
                    "samples": 10,
                    "seed": 42,
                    "num_steps": 3,
                },
            },
        },
        run_dir=tmp_path,
        context=context,
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )
    trainer.state["global_step"] = 1000

    trainer._validate()

    destination = tmp_path / "validation/step-00001000"
    losses = json.loads((destination / "loss-metrics.json").read_text())
    visual = json.loads((destination / "visual-report.json").read_text())
    assert losses["local_edit_keep_region_loss"] == 1.0
    assert losses["noop_full_image_loss"] == 2.0
    assert visual["selection_pool"] == "held_out_validation_sources"
    assert visual["sample_count"] == 10
    assert len(tuple((destination / "images").glob("*-panel.png"))) == 10
    assert "edit_region_change" in visual["aggregate"]
    assert "non_edit_region_preservation" in visual["aggregate"]
    assert "full_inference_image_quality" in visual["aggregate"]
