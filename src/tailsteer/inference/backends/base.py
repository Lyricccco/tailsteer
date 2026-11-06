"""Small boundary between shared experiment planning and native model runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class InferenceBuildContext:
    training_run: Path
    training_config_path: Path
    training_config: dict[str, Any]
    export_path: Path
    export_manifest: dict[str, Any]
    export_step: int
    suite_path: Path
    suite: dict[str, Any]
    output_dir: Path
    policies: dict[str, Any]
    inference_overrides: dict[str, Any]
    evaluation: dict[str, Any]
    execution: dict[str, Any] | None = None


class InferenceBackend(Protocol):
    key: str
    model_names: frozenset[str]

    def actor_step_count(self, num_steps: int) -> int:
        """Return the number of model forwards for a sampler timetable."""

    def forward_step_to_actor_index(self, step: int) -> int:
        """Map the cache/training forward label to a zero-based sampler index."""

    def validate_transition(
        self, *, start_step: int, supervision_step: int, num_steps: int
    ) -> None:
        """Validate model-native training boundaries for an inference schedule."""

    def build_config(self, context: InferenceBuildContext) -> dict[str, Any]:
        """Render the native runner config from one shared inference plan."""

    def run(self, config: dict[str, Any], *, resume: bool) -> dict[str, Any]:
        """Execute the native runner in the current torchrun worker."""
