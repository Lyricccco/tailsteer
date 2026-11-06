from __future__ import annotations


import pytest
import torch

from tailsteer.models.flux1_kontext import official


class Model(torch.nn.Module):
    def forward(self, **kwargs):
        return torch.ones_like(kwargs["img"])


def test_capture_observes_target_only_and_preserves_precision(monkeypatch):
    monkeypatch.setattr(
        official, "unpack_state", lambda x, **kw: x.detach().cpu().clone()
    )
    model = Model()
    state = torch.zeros(1, 4, 64, dtype=torch.bfloat16)
    source = torch.full_like(state, 7)
    sigmas = [1.0, 0.5, 0.0]
    with official.capture_states(
        model, points=(0, 1), sigmas=sigmas, height=32, width=32
    ) as trace:
        for k in range(2):
            velocity = model(
                img=torch.cat([state, source], 1),
                timesteps=torch.tensor([sigmas[k]], dtype=state.dtype),
            )
            state = state + (sigmas[k + 1] - sigmas[k]) * velocity[:, :4]
    assert torch.all(trace.forward_states[0] == 0)
    assert torch.all(trace.forward_states[1] == -0.5)
    assert trace.forward_states[1].dtype == torch.bfloat16
    assert trace.forward_states[0].shape == (1, 4, 64)
    assert trace.schedule_points == {0: (1.0, 1.0), 1: (0.5, 0.5)}
    assert not model._forward_pre_hooks


def test_capture_removes_hook_on_incomplete_trajectory():
    model = Model()
    with (
        pytest.raises(RuntimeError, match="incomplete"),
        official.capture_states(
            model, points=(0,), sigmas=[1.0, 0.0], height=32, width=32
        ),
    ):
        pass
    assert not model._forward_pre_hooks


@pytest.mark.parametrize("points", [(), (-1,), (2,), (1, 0), (0, 0)])
def test_invalid_points_rejected(points):
    with (
        pytest.raises(ValueError),
        official.capture_states(
            Model(), points=points, sigmas=[1.0, 0.5, 0.0], height=32, width=32
        ),
    ):
        pytest.fail("bad capture points accepted")
