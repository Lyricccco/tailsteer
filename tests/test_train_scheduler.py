from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


def _scheduler(total, schedule_steps=None):
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("tailsteer_train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = module._scheduler(
        optimizer, warmup=100, total=total, schedule_steps=schedule_steps
    )
    return optimizer, scheduler


def test_default_cosine_schedule_is_unchanged():
    _, scheduler = _scheduler(2000)
    scale = scheduler.lr_lambdas[0]
    for step in (0, 50, 99, 100, 500, 1000, 1999, 2000, 3000):
        expected = (
            (step + 1) / 100
            if step < 100
            else 0.5 * (1 + math.cos(math.pi * min((step - 100) / 1900, 1)))
        )
        assert scale(step) == pytest.approx(expected)


def test_slow_cosine_is_independent_of_training_duration():
    _, short = _scheduler(2000, 10000)
    _, long = _scheduler(5000, 10000)
    for step in range(2001):
        assert short.lr_lambdas[0](step) == long.lr_lambdas[0](step)
    assert 0.91 < short.lr_lambdas[0](2000) < 0.92
    assert 0.50 < long.lr_lambdas[0](5000) < 0.51
    assert long.lr_lambdas[0](10000) == 0


def test_slow_cosine_resume_matches_uninterrupted_training():
    optimizer, scheduler = _scheduler(5000, 10000)
    for _ in range(2000):
        optimizer.step()
        scheduler.step()
    restored_optimizer, restored_scheduler = _scheduler(5000, 10000)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()


@pytest.mark.parametrize("value", [0, -1, 100, True, 5000.0, "10000"])
def test_invalid_schedule_horizon_is_rejected(value):
    with pytest.raises(ValueError, match="lr_schedule_steps"):
        _scheduler(2000, value)
