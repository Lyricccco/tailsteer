"""Transparent PyTorch training and practical distributed resume."""

from tailsteer.cache.trajectory_dataset import CachedPathReference

from .distributed import DistributedContext, initialize_distributed, prepare_model
from .target_geometry import (
    CACHED_PATH,
    INSTANT_VELOCITY,
    TERMINAL_BRIDGE,
    TRACKING_PATH,
    TargetGeometry,
    TargetGeometryContext,
    TargetGeometryResult,
    build_target_geometry,
)
from .trainer import Trainer
from .start_state_augmentation import StartStateAugmentation

__all__ = [
    "CACHED_PATH",
    "INSTANT_VELOCITY",
    "TERMINAL_BRIDGE",
    "TRACKING_PATH",
    "CachedPathReference",
    "DistributedContext",
    "TargetGeometry",
    "TargetGeometryContext",
    "TargetGeometryResult",
    "Trainer",
    "StartStateAugmentation",
    "build_target_geometry",
    "initialize_distributed",
    "prepare_model",
]
