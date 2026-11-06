"""The four final recipes share one objective; only sampling/state sources differ."""

from __future__ import annotations

from collections.abc import Mapping

from .target_step_sampling import validate_sigma_beta_config

ALGORITHM = "trajectory_preservation"
RECIPES = ("a", "b", "c", "d")


def configure_recipe(config: dict, recipe: str) -> None:
    """Expand a common training config before applying experiment overrides."""
    if recipe not in RECIPES:
        raise ValueError(f"unknown trajectory-preservation recipe: {recipe}")
    training = config["training"]
    training["recipe"] = recipe
    transition = training["transition"]
    last = int(transition["supervision_step"])
    if recipe in {"a", "b"}:
        transition["start_step"] = last if recipe == "a" else last - 1
        transition["alpha_schedule"] = {"name": "constant", "value": 1.0}
        training["target_step_sampling"] = {
            "strategy": "fixed" if recipe == "a" else "uniform_tail"
        }
    if recipe in {"a", "c"}:
        training.pop("state_source", None)
        training.pop("rollout_start_state", None)
    elif recipe == "b":
        training["state_source"] = {"strategy": "fixed_tail"}


def validate_recipe(training: Mapping) -> None:
    """Reject combinations that would silently change the agreed comparisons."""
    recipe = training.get("recipe")
    if recipe not in RECIPES:
        raise ValueError("trajectory_preservation requires recipe a/b/c/d")
    transition = training["transition"]
    start, last = transition["start_step"], transition["supervision_step"]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, last)):
        raise TypeError("start_step and supervision_step must be integers")
    if start < 0 or start > last or (recipe in {"c", "d"} and start == last):
        raise ValueError("invalid recipe start/supervision steps")
    constant_one = transition["alpha_schedule"] == {
        "name": "constant",
        "value": 1.0,
    }
    if recipe == "a" and not constant_one:
        raise ValueError("A requires constant alpha=1")
    if recipe == "b" and not constant_one and start == last:
        raise ValueError(
            "B with a transition alpha requires start_step < supervision_step"
        )
    if recipe in {"c", "d"} and transition["alpha_schedule"].get("name") == "constant":
        raise ValueError("C/D require a transition alpha schedule")
    target = training["teacher_guided_target"]
    if target.get("loss") != "v-target":
        raise ValueError("final recipes require v-target supervision")
    geometries = (
        {"instant_velocity", "straight_tail"} if recipe == "b" else {"instant_velocity"}
    )
    if target.get("geometry") not in geometries:
        raise ValueError(
            "straight_tail geometry is a B-only ablation; "
            "otherwise use instant_velocity"
        )
    if target.get("geometry") == "straight_tail" and target.get("mode") != "composed":
        raise ValueError("straight_tail uses a single composed anchor direction")
    if target.get("geometry") == "straight_tail" and not constant_one:
        raise ValueError("B straight_tail requires constant alpha=1")
    if target.get("endpoint", {}).get("mode") != "source_keep_terminal_edit":
        raise ValueError("final recipes require source_keep_terminal_edit")
    if float(training.get("teacher_guided_weight", 1.0)) != 1.0:
        raise ValueError("final recipes use loss weight 1; tune velocity_scale instead")
    expected_sampling = {
        "a": {"fixed"},
        "b": {"uniform_tail", "sigma_beta"},
        "c": {"region_weighted", "sigma_beta"},
        "d": {"region_weighted", "sigma_beta"},
    }
    sampling = training["target_step_sampling"]
    if sampling.get("strategy") not in expected_sampling[recipe]:
        raise ValueError(
            f"recipe {recipe} requires sampling in {sorted(expected_sampling[recipe])}"
        )
    if sampling.get("strategy") == "sigma_beta":
        validate_sigma_beta_config(sampling)
        scope = "tail" if recipe == "b" else "all"
        if sampling.get("scope", "all") != scope:
            raise ValueError(f"recipe {recipe} requires sigma_beta scope={scope}")
    state = training.get("state_source")
    if recipe in {"a", "c"} and state is not None:
        raise ValueError("A/C use cached replay, without state_source rollout")
    if recipe == "b" and state not in (
        {"strategy": "fixed_tail"},
        {"strategy": "cached"},
    ):
        raise ValueError("B state_source must be fixed_tail or cached")
    if recipe == "d" and (
        not isinstance(state, Mapping)
        or state.get("strategy", "local_window")
        not in {"local_window", "cached_segment"}
    ):
        raise ValueError("D requires a local_window or cached_segment rollout")
