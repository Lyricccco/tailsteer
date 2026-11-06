"""Translate paper terminology into the numerical training implementation."""

from copy import deepcopy

BACKBONES = {
    "flux1-kontext-dev": "flux1_kontext_dev",
    "flux2-klein-base-9b": "flux2_klein_9b",
    "qwen-image-2.1": "qwen_image_21",
    "sensenova-u1.5-8b-mot": "sensenova_u1_5_8b_mot",
}


def normalize(config):
    config = deepcopy(config)
    training = config.get("training", {})
    if "steering_onset" not in training:
        return config
    onset = training.pop("steering_onset")
    last = training.pop("last_denoising_step")
    blending = training.pop("blending_schedule", "odds")
    odds = training.pop("odds_ratio", 1.0)
    training["transition"] = dict(
        start_step=onset,
        supervision_step=last,
        alpha_schedule=dict(name=blending, odds_ratio=odds),
    )
    # The implementation's alpha is preservation weight; paper lambda = 1-alpha.
    sampling = training.pop("timestep_sampling")
    training["target_step_sampling"] = dict(
        strategy=sampling["strategy"],
        uniform_mix=sampling["rho"],
        beta=sampling["beta"],
    )
    training["teacher_guided_target"] = dict(
        loss="v-target",
        mode="composed",
        geometry="instant_velocity",
        endpoint=dict(mode="source_keep_terminal_edit"),
        velocity_scale=1.0,
    )
    if "final_step_loss_weight" in training:
        training["teacher_guided_target"]["endpoint_loss_weight"] = training.pop(
            "final_step_loss_weight"
        )
    endpoint = training.pop("preservation_endpoint", "source_keep_terminal_edit")
    if endpoint != "source_keep_terminal_edit":
        raise ValueError("The paper configuration uses source_keep_terminal_edit")
    training.update(
        algorithm="trajectory_preservation",
        recipe="d",
        teacher_guided_weight=1.0,
        state_source=dict(strategy="cached_segment"),
        rollout_start_state="cached",
    )
    for public, internal in (
        ("max_updates", "max_steps"),
        ("warmup_updates", "warmup_steps"),
        ("lr_decay_updates", "lr_schedule_steps"),
        ("checkpoint_every_updates", "checkpoint_every"),
        ("validate_every_updates", "validate_every"),
        ("log_every_updates", "log_every"),
    ):
        if public in training:
            training[internal] = training.pop(public)
    return config
