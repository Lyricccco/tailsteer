from __future__ import annotations

from types import SimpleNamespace

import torch

from tailsteer.cache.sample import CacheSample
from tailsteer.training.distributed import DistributedContext
from tailsteer.training.trainer import Trainer, _requires_gradient_sync


class _Model:
    trainable_policy = "full"

    def __init__(self, parameter):
        self.parameter = parameter
        self.root = torch.nn.Linear(1, 1, bias=False)
        self.root.weight = parameter

    def get_trainable_module(self):
        return self.root

    def trainable_parameters(self):
        return (self.parameter,)


class _Algorithm:
    def __init__(self, parameter):
        self.parameter = parameter

    def compute_loss(self, batch):
        x = batch["latent"].float().flatten()
        target = batch["source_latent"].float().flatten()
        loss = (self.parameter.flatten() * x - target).square().mean()
        return SimpleNamespace(loss=loss, metrics={"loss": loss.detach()})


def _item(value: float):
    return CacheSample(
        latent=torch.tensor([[[value]]]),
        source_latent=torch.tensor([[[1.0]]]),
        keep_mask=torch.ones((1, 1, 1)),
        metadata={
            "sample_id": str(value),
            "source_id": str(value),
            "source_identity": str(value),
            "prompt": "x",
            "seed": 0,
            "step_index": 0,
            "sigma": 0.5,
            "timestep": 500.0,
            "height": 16,
            "width": 16,
            "task": "noop",
        },
    )


def test_gradient_accumulation_matches_manual_batch_mean(tmp_path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([[0.25]]))
    model = _Model(parameter)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    config = {
        "seed": 0,
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "num_workers": 0,
            "max_epochs": 1,
            "max_steps": None,
            "max_grad_norm": None,
            "log_every": 0,
            "validate_every": 0,
            "checkpoint_every": 0,
            "export_inference": False,
        },
    }
    context = DistributedContext("single", 0, 0, 1, torch.device("cpu"))
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    trainer = Trainer(
        model=model,
        algorithm=_Algorithm(parameter),
        optimizer=optimizer,
        scheduler=None,
        train_dataset=[_item(1.0), _item(2.0)],
        validation_dataset=[],
        config=config,
        run_dir=tmp_path,
        context=context,
        logger=logger,
    )
    # DataLoader shuffles in single mode; the mean gradient is order independent.
    trainer.train()

    reference = torch.nn.Parameter(torch.tensor([[0.25]]))
    reference_optimizer = torch.optim.SGD([reference], lr=0.1)
    reference_loss = sum((reference * value - 1.0).square() for value in (1.0, 2.0)) / 2
    reference_loss.backward()
    reference_optimizer.step()
    torch.testing.assert_close(parameter, reference)


def test_fsdp2_syncs_each_microbatch_but_ddp_waits_for_step_boundary() -> None:
    assert _requires_gradient_sync("fsdp2", optimizer_boundary=False)
    assert not _requires_gradient_sync("ddp", optimizer_boundary=False)
    assert not _requires_gradient_sync("single", optimizer_boundary=False)
    assert _requires_gradient_sync("fsdp2", optimizer_boundary=True)
    assert _requires_gradient_sync("ddp", optimizer_boundary=True)


def test_resumed_training_at_step_limit_does_not_take_an_extra_step(tmp_path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([[0.25]]))
    model = _Model(parameter)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    config = {
        "seed": 0,
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "max_epochs": None,
            "max_steps": 1,
            "max_grad_norm": None,
            "log_every": 0,
            "validate_every": 0,
            "checkpoint_every": 0,
            "export_inference": False,
        },
    }
    messages: list[tuple] = []
    trainer = Trainer(
        model=model,
        algorithm=_Algorithm(parameter),
        optimizer=optimizer,
        scheduler=None,
        train_dataset=[_item(1.0)],
        validation_dataset=[],
        config=config,
        run_dir=tmp_path,
        context=DistributedContext("single", 0, 0, 1, torch.device("cpu")),
        logger=SimpleNamespace(info=lambda *args, **_kwargs: messages.append(args)),
    )
    trainer.state = {"global_step": 1, "epoch": 0, "batch_in_epoch": 1}
    before = parameter.detach().clone()

    assert trainer.train() == trainer.state
    torch.testing.assert_close(parameter, before)
    assert any("target already reached" in str(message[0]) for message in messages)


def test_transient_stop_boundary_preserves_configured_horizon(tmp_path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([[0.25]]))
    model = _Model(parameter)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    config = {
        "seed": 0,
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "max_epochs": 1,
            "max_steps": 3,
            "max_grad_norm": None,
            "log_every": 0,
            "validate_every": 0,
            "checkpoint_every": 0,
            "export_inference": False,
        },
    }
    trainer = Trainer(
        model=model,
        algorithm=_Algorithm(parameter),
        optimizer=optimizer,
        scheduler=None,
        train_dataset=[_item(1.0), _item(2.0)],
        validation_dataset=[],
        config=config,
        run_dir=tmp_path,
        context=DistributedContext("single", 0, 0, 1, torch.device("cpu")),
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )

    state = trainer.train(stop_at_step=1)

    assert state["global_step"] == 1
    assert config["training"]["max_steps"] == 3
