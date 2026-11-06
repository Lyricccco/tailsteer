"""Model-neutral target-step sampling for teacher-guided training."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

REGION_NAMES = ("prior", "transition")
LEGACY_REGION_NAMES = ("prior", "transition", "endpoint")
DEFAULT_REGION_PROBABILITIES = {
    "prior": 0.25,
    "transition": 0.75,
}
DEFAULT_SIGMA_BETA_SAMPLING = {
    "strategy": "sigma_beta",
    "uniform_mix": 0.8,
    "beta": 3.0,
}


def validate_sigma_beta_config(config: Mapping[str, Any]) -> tuple[float, float]:
    """Validate the small sigma-only policy before loading any model weights."""
    if config.get("strategy") != "sigma_beta":
        raise ValueError("expected target_step_sampling.strategy = sigma_beta")
    unexpected = set(config) - set(DEFAULT_SIGMA_BETA_SAMPLING) - {"scope"}
    if unexpected:
        raise ValueError(f"sigma_beta does not accept fields: {sorted(unexpected)}")
    if config.get("scope", "all") not in {"all", "tail"}:
        raise ValueError("sigma_beta scope must be all or tail")
    values = []
    for key in ("uniform_mix", "beta"):
        value = config.get(key, DEFAULT_SIGMA_BETA_SAMPLING[key])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"sampling {key} must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"sampling {key} must be finite")
        values.append(float(value))
    rho, beta = values
    if not 0 <= rho <= 1 or beta <= 0:
        raise ValueError("sampling uniform_mix must be in [0, 1] and beta > 0")
    return rho, beta


def sigma_beta_probabilities(sigmas, config):
    """Integrate the shared density on a complete native noise-to-clean grid."""
    rho, beta = validate_sigma_beta_config(config)
    if len(sigmas) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in sigmas):
        raise ValueError("sigma_beta requires finite sigmas in [0, 1]")
    if not (
        math.isclose(sigmas[0], 1.0, abs_tol=1e-6, rel_tol=0)
        and math.isclose(sigmas[-1], 0.0, abs_tol=1e-6, rel_tol=0)
    ):
        raise ValueError(
            "sigma_beta requires the complete noise-to-clean sigma range [1, 0]"
        )
    if any(a <= b for a, b in zip(sigmas[:-1], sigmas[1:], strict=True)):
        raise ValueError("sigma_beta requires strictly decreasing sigmas")
    cdf = [rho * s + (1 - rho) * (1 - (1 - s) ** beta) for s in sigmas]
    masses = [a - b for a, b in zip(cdf[:-1], cdf[1:], strict=True)]
    total = math.fsum(masses)
    if any(p < 0 for p in masses) or not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("sigma_beta interval probabilities must sum to one")
    return tuple(p / total for p in masses)


class UniformStepRangeSampler:
    """Sample one model-forward step uniformly from an inclusive range.

    B's fixed tail target is intentionally independent of C/D's semantic
    prior/transition/endpoint probabilities.  Keeping this sampler here lets
    it share deterministic validation draws without coupling the algorithms.
    """

    def __init__(self, start_step: int, end_step: int, *, seed: int = 0) -> None:
        if isinstance(start_step, bool) or isinstance(end_step, bool):
            raise TypeError("uniform tail step bounds must be integers")
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        if self.start_step < 0 or self.start_step > self.end_step:
            raise ValueError(
                "uniform tail step range must satisfy 0 <= start_step <= end_step"
            )
        self.seed = int(seed)

    def sample(self, *, training: bool, key: str) -> int:
        if training:
            import torch

            draw = float(torch.rand(()).item())
        else:
            draw = RegionWeightedTargetStepSampler._unit_interval(
                self.seed, key, "tail"
            )
        count = self.end_step - self.start_step + 1
        index = min(int(draw * count), count - 1)
        return self.start_step + index


def validate_inference_schedule(
    contract: Any,
    schedule: Mapping[int, Any],
    *,
    supervision_step: int,
) -> None:
    """Require the live model schedule to reproduce cached trajectory points."""

    missing = []
    mismatches = []
    for cached in contract.forward_points:
        step = int(cached.step_index)
        if step > int(supervision_step):
            continue
        if step not in schedule:
            missing.append(step)
            continue
        live = schedule[step]
        if not (
            math.isclose(
                float(cached.sigma), float(live.sigma), rel_tol=0.0, abs_tol=1e-4
            )
            and math.isclose(
                float(cached.timestep),
                float(live.timestep),
                rel_tol=0.0,
                abs_tol=1e-3,
            )
        ):
            mismatches.append(
                (
                    step,
                    float(cached.sigma),
                    float(live.sigma),
                    float(cached.timestep),
                    float(live.timestep),
                )
            )
    if missing or mismatches:
        raise ValueError(
            "live inference schedule differs from trajectory cache: "
            f"missing={missing} mismatches={mismatches[:4]}"
        )


@dataclass(frozen=True)
class TargetStepSample:
    step: int
    region: str


class TailTargetStepSampler:
    """A's single cached point and B's uniform tail, sharing the C/D interface."""

    def __init__(self, transition, *, fixed: bool, seed: int = 0):
        start = int(transition["start_step"])
        end = start if fixed else int(transition["supervision_step"])
        self.steps = {"tail": tuple(range(start, end + 1))}
        self.sampler = UniformStepRangeSampler(start, end, seed=seed)

    def sample(self, *, training: bool, key: str) -> TargetStepSample:
        return TargetStepSample(self.sampler.sample(training=training, key=key), "tail")

    @staticmethod
    def metrics(sample, reference):
        return {"sampled_region_tail": reference.new_tensor(1.0)}


def build_target_step_sampler(
    contract, transition, config=None, *, seed=0, schedule=None
):
    strategy = (config or {}).get("strategy", "region_weighted")
    if strategy in {"fixed", "uniform_tail"}:
        return TailTargetStepSampler(transition, fixed=strategy == "fixed", seed=seed)
    if strategy == "sigma_beta":
        return SigmaBetaTargetStepSampler(
            contract, transition, config, schedule=schedule, seed=seed
        )
    return RegionWeightedTargetStepSampler(contract, transition, config, seed=seed)


class SigmaBetaTargetStepSampler:
    """Integrate one fixed sigma density over native forward Euler intervals.

    The cache supplies states, not sampling bins. With the default all scope,
    start_step only labels regions. B's explicit tail scope conditions this
    same distribution on start_step..supervision_step (inclusive). No importance
    correction is applied: this density defines the intended weighted loss.
    """

    def __init__(self, contract, transition, config, *, schedule, seed=0):
        self.uniform_mix, self.beta = validate_sigma_beta_config(config)
        self.scope = config.get("scope", "all")
        if not schedule:
            raise ValueError("sigma_beta requires the live model inference schedule")
        self.first_step = min(int(p.step_index) for p in contract.forward_points)
        self.start_step = int(transition["start_step"])
        self.supervision_step = int(transition["supervision_step"])
        if not self.first_step <= self.start_step < self.supervision_step:
            raise ValueError("invalid prior/transition/supervision step ordering")
        terminal = contract.terminal_point
        if int(terminal.step_index) != self.supervision_step + 1:
            raise ValueError("sigma_beta requires final-forward supervision")
        self.forward_steps = tuple(range(self.first_step, self.supervision_step + 1))
        missing = [
            k
            for k in (*self.forward_steps, self.supervision_step + 1)
            if k not in schedule
        ]
        if missing:
            raise ValueError(
                f"model schedule lacks sigma_beta interval points: {missing}"
            )
        self.sigmas = tuple(
            float(schedule[k].sigma)
            for k in (*self.forward_steps, self.supervision_step + 1)
        )
        if not math.isclose(
            float(terminal.sigma), self.sigmas[-1], abs_tol=1e-6, rel_tol=0
        ):
            raise ValueError("sigma_beta terminal sigma differs from schedule")
        self.step_probabilities = sigma_beta_probabilities(self.sigmas, config)
        if self.scope == "tail":
            offset = self.forward_steps.index(self.start_step)
            self.forward_steps = self.forward_steps[offset:]
            self.sigmas = self.sigmas[offset:]
            masses = self.step_probabilities[offset:]
            self.step_probabilities = tuple(p / math.fsum(masses) for p in masses)
        self.cumulative = []
        for probability in self.step_probabilities:
            self.cumulative.append(
                probability + (self.cumulative[-1] if self.cumulative else 0.0)
            )
        self.cumulative[-1] = 1.0
        self.steps = {
            "prior": tuple(k for k in self.forward_steps if k <= self.start_step),
            "transition": tuple(k for k in self.forward_steps if k > self.start_step),
        }
        self.seed = int(seed)

    def sample(self, *, training: bool, key: str) -> TargetStepSample:
        if training:
            import torch

            draw = float(torch.rand(()).item())
        else:
            draw = RegionWeightedTargetStepSampler._unit_interval(
                self.seed, key, "sigma_beta"
            )
        index = min(bisect_right(self.cumulative, draw), len(self.forward_steps) - 1)
        step = self.forward_steps[index]
        return TargetStepSample(
            step, "prior" if step <= self.start_step else "transition"
        )

    @staticmethod
    def metrics(sample, reference):
        return {
            f"sampled_region_{name}": reference.new_tensor(float(sample.region == name))
            for name in REGION_NAMES
        }

    def describe(self):
        rows = [
            {
                "step": step,
                "sigma": self.sigmas[i],
                "next_sigma": self.sigmas[i + 1],
                "probability": self.step_probabilities[i],
                "region": "prior" if step <= self.start_step else "transition",
            }
            for i, step in enumerate(self.forward_steps)
        ]
        return {
            "strategy": "sigma_beta",
            **({"scope": "tail"} if self.scope == "tail" else {}),
            "uniform_mix": self.uniform_mix,
            "beta": self.beta,
            "start_step": self.start_step,
            "supervision_step": self.supervision_step,
            "discretization": "F(sigma_k) - F(sigma_next); terminal not sampled",
            "region_probabilities": {
                name: math.fsum(
                    row["probability"] for row in rows if row["region"] == name
                )
                for name in REGION_NAMES
            },
            "steps": rows,
        }


class RegionWeightedTargetStepSampler:
    """Sample inference-grid steps from configurable semantic regions.

    Region boundaries come from the trajectory contract and transition config,
    so this policy is shared by every model backend.  Cache point density never
    affects the target-step probability.
    """

    def __init__(
        self,
        contract: Any,
        transition: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        config = {} if config is None else dict(config)
        strategy = str(config.get("strategy", "region_weighted"))
        if strategy != "region_weighted":
            raise ValueError("target_step_sampling.strategy must be 'region_weighted'")
        coordinate = str(config.get("coordinate", "inference_step"))
        if coordinate != "inference_step":
            raise ValueError("target_step_sampling.coordinate must be 'inference_step'")

        forward_steps = tuple(
            int(point.step_index) for point in contract.forward_points
        )
        if not forward_steps:
            raise ValueError("target-step sampling requires forward cache points")
        self.first_step = min(forward_steps)
        self.start_step = int(transition["start_step"])
        self.supervision_step = int(transition["supervision_step"])
        if not self.first_step <= self.start_step < self.supervision_step:
            raise ValueError("invalid prior/transition/supervision step ordering")

        self.steps = {
            "prior": tuple(range(self.first_step, self.start_step + 1)),
            "transition": tuple(range(self.start_step + 1, self.supervision_step + 1)),
        }
        raw_probabilities = config.get(
            "region_probabilities", DEFAULT_REGION_PROBABILITIES
        )
        if not isinstance(raw_probabilities, Mapping):
            raise TypeError("target-step region_probabilities must be a mapping")
        configured_names = set(raw_probabilities)
        if configured_names == set(LEGACY_REGION_NAMES):
            # Preserve old resolved experiments exactly. New configurations
            # omit the separate endpoint bucket and include supervision_step
            # in transition instead.
            region_names = LEGACY_REGION_NAMES
            self.steps = {
                "prior": tuple(range(self.first_step, self.start_step + 1)),
                "transition": tuple(range(self.start_step + 1, self.supervision_step)),
                "endpoint": (self.supervision_step,),
            }
        elif configured_names == set(REGION_NAMES):
            region_names = REGION_NAMES
        else:
            expected = f"{REGION_NAMES} or {LEGACY_REGION_NAMES}"
            raise ValueError(
                "target-step region_probabilities require exactly "
                f"{expected}; actual={sorted(configured_names)}"
            )
        probabilities = {name: float(raw_probabilities[name]) for name in region_names}
        if any(
            not math.isfinite(value) or value < 0 for value in probabilities.values()
        ):
            raise ValueError(
                "target-step probabilities must be finite and non-negative"
            )
        total = sum(probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"target-step region probabilities must sum to 1, got {total}"
            )
        for name, probability in probabilities.items():
            if probability > 0 and not self.steps[name]:
                raise ValueError(
                    f"target-step region {name!r} is empty but has positive probability"
                )
        self.probabilities = probabilities
        self.region_names = region_names
        self.seed = int(seed)

    @staticmethod
    def _unit_interval(seed: int, key: str, stream: str) -> float:
        digest = hashlib.sha256(f"{seed}:{stream}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def _draws(self, *, training: bool, key: str) -> tuple[float, float]:
        if not training:
            return (
                self._unit_interval(self.seed, key, "region"),
                self._unit_interval(self.seed, key, "step"),
            )
        import torch

        return float(torch.rand(()).item()), float(torch.rand(()).item())

    def sample(self, *, training: bool, key: str) -> TargetStepSample:
        region_draw, step_draw = self._draws(training=training, key=key)
        cumulative = 0.0
        region = self.region_names[-1]
        for name in self.region_names:
            cumulative += self.probabilities[name]
            if region_draw < cumulative:
                region = name
                break
        candidates = self.steps[region]
        index = min(int(step_draw * len(candidates)), len(candidates) - 1)
        return TargetStepSample(step=candidates[index], region=region)

    def metrics(self, sample: TargetStepSample, reference: Any) -> dict[str, Any]:
        return {
            f"sampled_region_{name}": reference.new_tensor(float(sample.region == name))
            for name in self.region_names
        }
