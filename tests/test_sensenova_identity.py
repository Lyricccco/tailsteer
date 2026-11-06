from __future__ import annotations

import pytest

from tailsteer.models.sensenova.identity import (
    SENSENOVA_MODEL_NAMES,
    resolve_sensenova_variant,
)


def test_sensenova_release_identity() -> None:
    assert SENSENOVA_MODEL_NAMES == {"sensenova_u1_5_8b_mot"}
    assert resolve_sensenova_variant("sensenova_u1_5_8b_mot") == "sensenova-u1.5-8b-mot"


def test_sensenova_identity_rejects_cross_variant_configuration() -> None:
    with pytest.raises(ValueError, match="requires variant"):
        resolve_sensenova_variant(
            "sensenova_u1_5_8b_mot",
            "sensenova-u1.5-8b-mot-preview",
        )
