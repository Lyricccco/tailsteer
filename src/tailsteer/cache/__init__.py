"""Local, resumable native cache format and datasets."""

from .continuation import (
    CONTINUATION_MODE,
    prepare_cache_continuation,
    validate_cache_continuation,
)
from .dataset import (
    CacheDataset,
    cache_collate,
    compose_cache_datasets,
    open_cache_dataset,
)
from .finalize import finalize_trajectory_staging, inspect_trajectory_staging
from .format import (
    STEP_RECORDS_LAYOUT,
    TRAJECTORY_GROUPED_LAYOUT,
    CacheExample,
    CacheWriter,
    TrajectoryCacheWriter,
    read_cache_records,
    validate_cache,
)
from .sample import CacheSample
from .selection import (
    SELECTION_MODE,
    export_cache_selection,
    validate_cache_selection,
)
from .split import source_partition
from .tail_anchor_view import (
    TailAnchorViewDataset,
    canonicalize_tail_anchor_view,
    validate_tail_anchor_view,
)
from .trajectory import (
    TrajectoryResult,
    build_trajectory_cache,
    capture_trajectory,
    load_trajectory_runtime,
    validate_trajectory_config,
)
from .trajectory_dataset import (
    CachedPathReference,
    CachedPathReferenceResolver,
    CachedTrajectoryPoint,
    TrajectoryContract,
    TrajectorySegmentDataset,
    TrajectorySegmentSample,
    TrajectorySourceDataset,
    TrajectorySourceSample,
    compose_trajectory_segment_datasets,
    compose_trajectory_source_datasets,
    materialize_trajectory_state,
    read_trajectory_contract,
    trajectory_segment_collate,
    trajectory_source_collate,
)

__all__ = [
    "CacheDataset",
    "CacheExample",
    "CacheSample",
    "CacheWriter",
    "CONTINUATION_MODE",
    "SELECTION_MODE",
    "STEP_RECORDS_LAYOUT",
    "TailAnchorViewDataset",
    "TRAJECTORY_GROUPED_LAYOUT",
    "TrajectoryCacheWriter",
    "CachedTrajectoryPoint",
    "CachedPathReference",
    "CachedPathReferenceResolver",
    "TrajectoryContract",
    "TrajectorySegmentDataset",
    "TrajectorySegmentSample",
    "TrajectorySourceDataset",
    "TrajectorySourceSample",
    "TrajectoryResult",
    "build_trajectory_cache",
    "capture_trajectory",
    "export_cache_selection",
    "finalize_trajectory_staging",
    "inspect_trajectory_staging",
    "validate_trajectory_config",
    "cache_collate",
    "canonicalize_tail_anchor_view",
    "compose_cache_datasets",
    "compose_trajectory_segment_datasets",
    "compose_trajectory_source_datasets",
    "materialize_trajectory_state",
    "load_trajectory_runtime",
    "open_cache_dataset",
    "prepare_cache_continuation",
    "read_cache_records",
    "read_trajectory_contract",
    "source_partition",
    "validate_cache",
    "validate_cache_continuation",
    "validate_cache_selection",
    "validate_tail_anchor_view",
    "trajectory_segment_collate",
    "trajectory_source_collate",
]
