import dask.dataframe as dd
import dataclasses as dc
import pandas as pd
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from .constants import HUMANIZED_METRICS, HUMANIZED_VIEW_TYPES, Layer


class Score(Enum):
    TRIVIAL = 'trivial'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


Metric = str
ViewType = Literal['file_name', 'host_name', 'proc_name', 'step', 'time_range']
ViewKey = Union[
    Tuple[ViewType],
    Tuple[ViewType, ViewType],
    Tuple[ViewType, ViewType, ViewType],
    Tuple[ViewType, ViewType, ViewType, ViewType],
]


@dc.dataclass
class AnalysisRuntimeConfig:
    checkpoint: bool
    cluster_type: str
    debug: bool
    memory: int
    num_threads_per_worker: int
    num_workers: int
    processes: bool
    threshold: float
    verbose: bool
    working_dir: str


@dc.dataclass
class RawStats:
    job_time: "dd.Scalar"
    time_granularity: int
    time_resolution: int
    total_count: "dd.Scalar"


@dc.dataclass
class RuleReason:
    condition: str
    message: str


@dc.dataclass
class Rule:
    name: str
    condition: str
    layers: Optional[List[Layer]] = None
    reasons: Optional[List[RuleReason]] = None


@dc.dataclass
class RuleResultReason:
    description: str
    # value: Optional[float]


@dc.dataclass
class RuleResult:
    description: str
    compact_desc: Optional[str] = None
    detail_list: Optional[List[str]] = None
    extra_data: Optional[dict] = None
    object_hash: Optional[int] = None
    reasons: Optional[List[RuleResultReason]] = None
    value: Optional[Union[float, int, tuple]] = None
    value_fmt: Optional[str] = None


@dc.dataclass
class ViewResult:
    critical_view: dd.DataFrame
    metric: str
    records: dd.DataFrame
    view: dd.DataFrame
    view_type: ViewType


MainIndex = dd.DataFrame
MainView = dd.DataFrame
View = dd.DataFrame

Characteristics = Dict[str, RuleResult]
MetricBoundary = Union[int, float]
MetricBoundaries = Dict[str, Any]
ViewMetricBoundaries = Dict[str, MetricBoundaries]
Views = Dict[ViewKey, View]


@dc.dataclass
class OutputCharacteristicsType:
    complexity: float
    io_time: float
    job_time: float
    num_apps: int
    num_files: int
    num_nodes: int
    num_ops: int
    num_procs: int
    num_time_periods: int
    per_io_time: float


@dc.dataclass
class OutputCountsType:
    raw_count: int
    hlm_count: int
    main_view_count: int
    avg_perspective_count: Dict[str, int]
    avg_perspective_count_std: Dict[str, float]
    avg_perspective_critical_count: Dict[str, int]
    avg_perspective_critical_count_std: Dict[str, float]
    perspective_skewness: Dict[str, float]
    root_perspective_skewness: Dict[str, float]
    per_records_discarded: Dict[str, float]
    per_records_retained: Dict[str, float]
    num_metrics: int
    num_perspectives: int
    num_rules: int
    evaluated_records: Dict[str, int]
    perspective_count_tree: Dict[str, Dict[str, int]]
    perspective_critical_count_tree: Dict[str, Dict[str, int]]
    perspective_record_count_tree: Dict[str, Dict[str, int]]
    reasoned_records: Dict[str, int]
    slope_filtered_records: Dict[str, int]


@dc.dataclass
class OutputSeveritiesType:
    critical_count: Dict[str, int]
    critical_tree: Dict[str, Dict[str, int]]
    very_high_count: Dict[str, int]
    very_high_tree: Dict[str, Dict[str, int]]
    high_count: Dict[str, int]
    high_tree: Dict[str, Dict[str, int]]
    medium_count: Dict[str, int]
    medium_tree: Dict[str, Dict[str, int]]
    low_count: Dict[str, int]
    very_low_count: Dict[str, int]
    trivial_count: Dict[str, int]
    none_count: Dict[str, int]
    root_critical_count: Dict[str, int]
    root_very_high_count: Dict[str, int]
    root_high_count: Dict[str, int]
    root_medium_count: Dict[str, int]
    root_low_count: Dict[str, int]
    root_very_low_count: Dict[str, int]
    root_trivial_count: Dict[str, int]
    root_none_count: Dict[str, int]


@dc.dataclass
class OutputThroughputsType:
    evaluated_records: Dict[str, float]
    perspectives: Dict[str, float]
    reasoned_records: Dict[str, float]
    rules: Dict[str, float]
    slope_filtered_records: Dict[str, float]


@dc.dataclass
class OutputTimingsType:
    read_traces: Dict[str, float]
    compute_hlm: Dict[str, float]
    compute_main_view: Dict[str, float]
    compute_perspectives: Dict[str, float]
    attach_reasons: Dict[str, float]


@dc.dataclass
class OutputType:
    _characteristics: Characteristics
    _raw_stats: RawStats
    characteristics: OutputCharacteristicsType
    counts: OutputCountsType
    severities: OutputSeveritiesType
    throughputs: OutputThroughputsType
    timings: OutputTimingsType


@dc.dataclass
class AnalyzerResultType:
    _hlms: Dict[Layer, dd.DataFrame]
    _main_views: Dict[Layer, MainView]
    _metric_boundaries: ViewMetricBoundaries
    _traces: Optional[dd.DataFrame]
    checkpoint_dir: str
    flat_views: Dict[ViewKey, pd.DataFrame]
    layers: List[Layer]
    raw_stats: RawStats
    view_types: List[ViewType]
    views: Dict[Layer, Views]


def humanized_metric_name(metric: Metric):
    return HUMANIZED_METRICS[metric]


def humanized_view_name(view_key_type: Union[ViewKey, ViewType], separator='_'):
    if isinstance(view_key_type, tuple):
        return separator.join([HUMANIZED_VIEW_TYPES[view_type] for view_type in view_key_type])
    return HUMANIZED_VIEW_TYPES[view_key_type]


def view_name(view_key_type: Union[ViewKey, ViewType], separator='_'):
    return separator.join(view_key_type) if isinstance(view_key_type, tuple) else view_key_type
