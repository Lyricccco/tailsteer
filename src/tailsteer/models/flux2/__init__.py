"""FLUX.2 Klein Base integration."""

from .model import (
    Flux2Conditioning,
    Flux2Model,
    Flux2Schedule,
    SchedulePoint,
    build_flux2_schedule,
)
from .objective import Flux2ObjectiveResult, flux2_preservation_velocity
from .output_adapter import (
    Flux2OutputAdapterConfig,
    attach_flux2_output_adapter,
    build_residual_output_projection,
)
from .parity import Flux2OfficialTrace, capture_official_flux2_trace
from .policy import (
    FLUX2_STRUCTURED_POLICIES,
    Flux2PolicySelection,
    canonical_flux2_policy,
    configure_flux2_structured_trainable,
)
from .preprocessing import (
    FLUX2_MULTI_REFERENCE_MAX_PIXELS,
    FLUX2_REFERENCE_MULTIPLE,
    FLUX2_SINGLE_REFERENCE_MAX_PIXELS,
    flux2_reference_max_pixels,
    prepare_flux2_reference_image,
)
from .training import Flux2TrainingBatch, prepare_flux2_training_batch
from .vae import Flux2VaeDecoder

__all__ = [
    "Flux2Conditioning",
    "Flux2Model",
    "Flux2OfficialTrace",
    "Flux2ObjectiveResult",
    "Flux2TrainingBatch",
    "Flux2VaeDecoder",
    "Flux2PolicySelection",
    "Flux2OutputAdapterConfig",
    "FLUX2_STRUCTURED_POLICIES",
    "FLUX2_MULTI_REFERENCE_MAX_PIXELS",
    "FLUX2_REFERENCE_MULTIPLE",
    "FLUX2_SINGLE_REFERENCE_MAX_PIXELS",
    "Flux2Schedule",
    "SchedulePoint",
    "build_flux2_schedule",
    "capture_official_flux2_trace",
    "canonical_flux2_policy",
    "configure_flux2_structured_trainable",
    "attach_flux2_output_adapter",
    "build_residual_output_projection",
    "flux2_preservation_velocity",
    "flux2_reference_max_pixels",
    "prepare_flux2_reference_image",
    "prepare_flux2_training_batch",
]
