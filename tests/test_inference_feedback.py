from __future__ import annotations

import pytest

from tailsteer.inference.feedback import feedback_config, feedback_mode


def test_feedback_uses_one_canonical_mapping_and_reads_legacy_strings() -> None:
    assert feedback_config("model-native") == {"mode": "model_native"}
    assert feedback_mode({"mode": "continuous_image"}, default="image") == (
        "continuous_image"
    )
    assert feedback_mode("latent", default="image") == "latent"


def test_feedback_rejects_silent_extra_fields() -> None:
    with pytest.raises(ValueError, match="unsupported feedback fields"):
        feedback_mode({"mode": "latent", "output_decode": "immediate"}, default="image")
