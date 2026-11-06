from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from tailsteer.cache.trajectory_dataset import (
    CachedTrajectoryPoint,
    TrajectoryContract,
)
from tailsteer.training.target_step_sampling import (
    DEFAULT_SIGMA_BETA_SAMPLING,
    RegionWeightedTargetStepSampler,
    SigmaBetaTargetStepSampler,
    build_target_step_sampler,
    validate_sigma_beta_config,
)


def _contract() -> TrajectoryContract:
    return TrajectoryContract(
        model="toy",
        representation="toy",
        points=(
            CachedTrajectoryPoint(0, 1.0, 0.0, "forward_input"),
            CachedTrajectoryPoint(4, 0.5, 0.5, "forward_input"),
            CachedTrajectoryPoint(6, 0.2, 0.8, "forward_input"),
            CachedTrajectoryPoint(7, 0.1, 0.9, "forward_input"),
            CachedTrajectoryPoint(8, 0.0, 1.0, "final"),
        ),
        num_steps=8,
        height=16,
        width=16,
    )


def test_region_steps_come_from_model_contract_not_cache_density() -> None:
    sampler = RegionWeightedTargetStepSampler(
        _contract(), {"start_step": 4, "supervision_step": 7}
    )

    assert sampler.steps == {
        "prior": (0, 1, 2, 3, 4),
        "transition": (5, 6, 7),
    }
    assert sampler.probabilities == {
        "prior": 0.25,
        "transition": 0.75,
    }


@pytest.mark.parametrize("prior_probability", [0.25, 0.75])
def test_validation_sampling_is_stable_and_approximately_weighted(
    prior_probability,
) -> None:
    sampler = RegionWeightedTargetStepSampler(
        _contract(),
        {"start_step": 4, "supervision_step": 7},
        {
            "region_probabilities": {
                "prior": prior_probability,
                "transition": 1 - prior_probability,
            }
        },
        seed=17,
    )
    first = [sampler.sample(training=False, key=f"source-{i}") for i in range(4000)]
    second = [sampler.sample(training=False, key=f"source-{i}") for i in range(4000)]

    assert first == second
    regions = Counter(item.region for item in first)
    assert regions["prior"] / len(first) == pytest.approx(prior_probability, abs=0.025)
    assert regions["transition"] / len(first) == pytest.approx(
        1 - prior_probability, abs=0.025
    )
    transition_steps = Counter(
        item.step for item in first if item.region == "transition"
    )
    assert transition_steps[5] / sum(transition_steps.values()) == pytest.approx(
        1 / 3, abs=0.04
    )


def test_two_region_sampling_includes_supervision_step_at_start48() -> None:
    sampler = RegionWeightedTargetStepSampler(
        _contract(), {"start_step": 4, "supervision_step": 7}
    )
    assert sampler.steps["transition"][-1] == 7
    short_tail = RegionWeightedTargetStepSampler(
        _contract(), {"start_step": 6, "supervision_step": 7}
    )
    assert short_tail.steps == {
        "prior": (0, 1, 2, 3, 4, 5, 6),
        "transition": (7,),
    }


def test_explicit_three_region_sampling_keeps_legacy_boundaries() -> None:
    sampler = RegionWeightedTargetStepSampler(
        _contract(),
        {"start_step": 4, "supervision_step": 7},
        {
            "region_probabilities": {
                "prior": 0.25,
                "transition": 0.50,
                "endpoint": 0.25,
            }
        },
    )
    assert sampler.steps == {
        "prior": (0, 1, 2, 3, 4),
        "transition": (5, 6),
        "endpoint": (7,),
    }


def test_region_probabilities_must_form_a_complete_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        RegionWeightedTargetStepSampler(
            _contract(),
            {"start_step": 4, "supervision_step": 7},
            {
                "region_probabilities": {
                    "prior": 0.25,
                    "transition": 0.25,
                    "endpoint": 0.25,
                }
            },
        )


def _native_sigma_contract(model="flux2"):
    from tailsteer.models.flux2.model import Flux2Model
    from tailsteer.models.protocols import TrajectoryPoint

    if model == "flux2":
        schedule = Flux2Model.tail_schedule(None, num_steps=50, height=1024, width=1024)
        first, terminal = 0, 50
    else:
        first, terminal = 1, 51
        schedule = {}
        for offset in range(51):
            u = 1 - offset / 50
            sigma = 3 * u / (1 + 2 * u)
            step = offset + 1
            schedule[step] = TrajectoryPoint(step, sigma, 1 - sigma)
    cached = (*range(first, first + 40, 3), *range(first + 40, terminal + 1))
    contract = TrajectoryContract(
        model=model,
        representation="toy",
        num_steps=50,
        height=1024,
        width=1024,
        points=tuple(
            CachedTrajectoryPoint(
                k,
                schedule[k].sigma,
                schedule[k].timestep,
                "final" if k == terminal else "forward_input",
            )
            for k in cached
        ),
    )
    return contract, schedule


def _sigma_sampler(model="flux2", start=46, **options):
    contract, schedule = _native_sigma_contract(model)
    return build_target_step_sampler(
        contract,
        {
            "start_step": start,
            "supervision_step": contract.terminal_point.step_index - 1,
        },
        {**DEFAULT_SIGMA_BETA_SAMPLING, **options},
        schedule=schedule,
        seed=17,
    )


def test_tail_beta_is_conditioned_full_distribution_including_boundary():
    full = _sigma_sampler()
    tail = _sigma_sampler(scope="tail")
    assert tail.forward_steps == (46, 47, 48, 49)
    assert tail.steps == {"prior": (46,), "transition": (47, 48, 49)}
    total = sum(full.step_probabilities[46:])
    assert tail.step_probabilities == pytest.approx(
        tuple(p / total for p in full.step_probabilities[46:]), abs=1e-12
    )
    assert tail.step_probabilities == pytest.approx(
        (0.1570725, 0.2013627, 0.2678462, 0.3737186), abs=1e-6
    )
    assert {tail.sample(training=False, key=str(i)).step for i in range(1000)} == {
        46,
        47,
        48,
        49,
    }
    rows = tail.describe()["steps"]
    assert rows[0]["sigma"] == full.describe()["steps"][46]["sigma"]
    assert rows[-1]["next_sigma"] == 0


@pytest.mark.parametrize(
    "model,start,prior,last",
    [
        ("flux2", 46, 0.60087265, 0.17695628),
        ("sensenova", 42, 0.56063110, 0.07881059),
    ],
)
def test_sigma_beta_matches_native_schedule_and_analysis(model, start, prior, last):
    sampler = _sigma_sampler(model, start)
    assert isinstance(sampler, SigmaBetaTargetStepSampler)
    assert sum(sampler.step_probabilities) == pytest.approx(1, abs=1e-12)
    assert all(p >= 0 for p in sampler.step_probabilities)
    assert all(
        a < b
        for a, b in zip(
            sampler.step_probabilities[:-1], sampler.step_probabilities[1:], strict=True
        )
    )
    assert sampler.step_probabilities[-1] == pytest.approx(last, abs=1e-6)
    report = sampler.describe()
    assert report["region_probabilities"]["prior"] == pytest.approx(prior, abs=1e-6)
    assert report["steps"][-1]["next_sigma"] == 0
    assert len(report["steps"]) == 50


@pytest.mark.parametrize(
    "model,starts", [("flux2", (43, 46, 48)), ("sensenova", (37, 42, 47))]
)
def test_sigma_beta_is_independent_of_start_step_and_cache_density(model, starts):
    samplers = [_sigma_sampler(model, start) for start in starts]
    assert all(s.step_probabilities == samplers[0].step_probabilities for s in samplers)
    draws = [
        [s.sample(training=False, key=str(i)).step for i in range(200)]
        for s in samplers
    ]
    assert draws[0] == draws[1] == draws[2]
    contract, schedule = _native_sigma_contract(model)
    dense = replace(
        contract,
        points=tuple(
            CachedTrajectoryPoint(
                k,
                v.sigma,
                v.timestep,
                "final" if k == contract.terminal_point.step_index else "forward_input",
            )
            for k, v in schedule.items()
        ),
    )
    sampler = build_target_step_sampler(
        dense,
        {
            "start_step": starts[0],
            "supervision_step": dense.terminal_point.step_index - 1,
        },
        DEFAULT_SIGMA_BETA_SAMPLING,
        schedule=schedule,
    )
    assert sampler.step_probabilities == samplers[0].step_probabilities


@pytest.mark.parametrize(
    "config", [{"uniform_mix": 1}, {"beta": 1}, {"uniform_mix": 0, "beta": 1}]
)
def test_sigma_uniform_limit_means_interval_width_not_uniform_steps(config):
    sampler = _sigma_sampler(**config)
    assert sampler.step_probabilities == pytest.approx(
        [a - b for a, b in zip(sampler.sigmas[:-1], sampler.sigmas[1:], strict=True)],
        abs=1e-12,
    )


def test_sigma_beta_empirical_draws_and_validation_do_not_use_global_rng():
    import torch

    sampler = _sigma_sampler()
    torch.manual_seed(5)
    before = torch.get_rng_state().clone()
    deterministic = [sampler.sample(training=False, key=str(i)) for i in range(100)]
    assert torch.equal(before, torch.get_rng_state())
    assert deterministic == [
        sampler.sample(training=False, key=str(i)) for i in range(100)
    ]
    counts = Counter(
        sampler.sample(training=True, key="unused").step for _ in range(30000)
    )
    assert set(counts) == set(range(50))  # No terminal, and uncached steps are sampled.
    for k, probability in zip(
        sampler.forward_steps, sampler.step_probabilities, strict=True
    ):
        assert counts[k] / 30000 == pytest.approx(probability, abs=0.008)


@pytest.mark.parametrize(
    "options",
    [
        {"uniform_mix": -0.1},
        {"uniform_mix": 1.1},
        {"uniform_mix": float("nan")},
        {"beta": 0},
        {"beta": -1},
        {"beta": float("inf")},
        {"beta": True},
        {"region_probabilities": {"prior": 0.25, "transition": 0.75}},
    ],
)
def test_sigma_beta_rejects_invalid_parameters_and_region_mix(options):
    with pytest.raises((TypeError, ValueError)):
        validate_sigma_beta_config({**DEFAULT_SIGMA_BETA_SAMPLING, **options})


@pytest.mark.parametrize(
    "failure",
    ["missing", "partial", "ascending", "nan", "terminal", "early_supervision"],
)
def test_sigma_beta_requires_complete_native_schedule(failure):
    contract, schedule = _native_sigma_contract()
    transition = {"start_step": 46, "supervision_step": 49}
    if failure == "missing":
        schedule.pop(12)
    elif failure == "partial":
        schedule[0] = replace(schedule[0], sigma=0.99)
    elif failure == "ascending":
        schedule[10] = replace(schedule[10], sigma=schedule[9].sigma)
    elif failure == "nan":
        schedule[10] = replace(schedule[10], sigma=float("nan"))
    elif failure == "terminal":
        schedule[50] = replace(schedule[50], sigma=0.01)
    else:
        transition["supervision_step"] = 48
    with pytest.raises(ValueError):
        build_target_step_sampler(
            contract, transition, DEFAULT_SIGMA_BETA_SAMPLING, schedule=schedule
        )
