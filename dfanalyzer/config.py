import dataclasses as dc
import logging
import socket
from hydra.core.config_store import ConfigStore
from hydra.conf import HelpConf, JobConf
from omegaconf import MISSING
from typing import Any, Dict, List, Optional

from .constants import COL_TIME_RANGE, VIEW_TYPES
from .types import ViewMetricBoundaries
from .utils.env_utils import get_bool_env_var


CHECKPOINT_VIEWS = get_bool_env_var("DFANALYZER_CHECKPOINT_VIEWS", False)
DERIVED_POSIX_METRICS = {
    'data': 'io_cat == 1 or io_cat == 2',
    'read': 'io_cat == 1',
    'write': 'io_cat == 2',
    'metadata': 'io_cat == 3',
    'close': 'io_cat == 3 and func_name.str.contains("close") and ~func_name.str.contains("dir")',
    'open': 'io_cat == 3 and func_name.str.contains("open") and ~func_name.str.contains("dir")',
    'seek': 'io_cat == 3 and func_name.str.contains("seek")',
    'stat': 'io_cat == 3 and func_name.str.contains("stat")',
    'other': 'io_cat == 6',
    'sync': 'io_cat == 7',
}
HASH_CHECKPOINT_NAMES = get_bool_env_var("DFANALYZER_HASH_CHECKPOINT_NAMES", False)


@dc.dataclass
class AnalyzerPresetConfig:
    additional_metrics: Optional[Dict[str, Optional[str]]] = dc.field(default_factory=dict)
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(default_factory=dict)
    layer_defs: Dict[str, Optional[str]] = MISSING
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(default_factory=dict)
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(default_factory=dict)
    name: str = MISSING
    threaded_layers: Optional[List[str]] = dc.field(default_factory=list)
    unscored_metrics: Optional[List[str]] = dc.field(default_factory=list)


@dc.dataclass
class AnalyzerPresetConfigPOSIX(AnalyzerPresetConfig):
    additional_metrics: Optional[Dict[str, Optional[str]]] = dc.field(default_factory=dict)
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {
            'posix': DERIVED_POSIX_METRICS,
        }
    )
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'posix': 'cat.str.contains("posix|stdio")',
        }
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(default_factory=dict)
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(
        default_factory=lambda: {
            'file_name': {
                'file_dir': None,
                'file_pattern': None,
            },
            'proc_name': {
                'host_name': 'proc_name.str.split("#").str[1]',
                'proc_id': 'proc_name.str.split("#").str[2]',
                'thread_id': 'proc_name.str.split("#").str[3]',
            },
        }
    )
    name: str = "posix"
    threaded_layers: Optional[List[str]] = dc.field(default_factory=list)
    unscored_metrics: Optional[List[str]] = dc.field(default_factory=list)


@dc.dataclass
class AnalyzerPresetConfigDLIO(AnalyzerPresetConfig):
    additional_metrics: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: {
            # 'compute_avg_througput':
            # 'compute_util': 'compute_{time_metric}.fillna(0) / (compute_{time_metric}.fillna(0) + fetch_data_{time_metric}.fillna(0) + checkpoint_{time_metric}.fillna(0))',
            'compute_util': 'compute_{time_metric} / (app_{time_metric} + {epsilon})',
            'fetch_data_util': 'fetch_data_{time_metric} / (app_{time_metric} + {epsilon})',
            'checkpoint_util': 'checkpoint_{time_metric} / (app_{time_metric} + {epsilon})',
            # 'consumer_rate': 'data_loader_item_count_sum / compute_time_sum',
            # 'producer_rate': 'data_loader_item_count_sum / data_loader_item_time_sum',
            # 'producer_consumer_rate': 'producer_rate / consumer_rate',
        }
    )
    derived_metrics: Optional[Dict[str, Dict[str, str]]] = dc.field(
        default_factory=lambda: {
            'app': {},
            'training': {},
            'compute': {},
            'fetch_data': {},
            'data_loader': {
                'init': 'func_name.str.contains("init")',
                'item': 'func_name.str.contains("__getitem__")',
            },
            'data_loader_fork': {},
            'reader': {
                'close': 'func_name.str.contains(".close")',
                'open': 'func_name.str.contains(".open")',  # e.g. NPZReader.open
                'preprocess': 'func_name.str.contains(".preprocess")',
                'sample': 'func_name.str.contains(".get_sample")',
            },
            # 'reader_posix': DERIVED_POSIX_METRICS,
            'reader_posix_lustre': DERIVED_POSIX_METRICS,
            # 'reader_posix_ssd': DERIVED_POSIX_METRICS,
            'checkpoint': {},
            # 'checkpoint_posix': {},
            'checkpoint_posix_lustre': DERIVED_POSIX_METRICS,
            'checkpoint_posix_ssd': DERIVED_POSIX_METRICS,
            'other_posix': DERIVED_POSIX_METRICS,
            # 'other_posix_lustre': DERIVED_POSIX_METRICS,
            # 'other_posix_ssd': DERIVED_POSIX_METRICS,
        }
    )
    layer_defs: Dict[str, Optional[str]] = dc.field(
        default_factory=lambda: {
            'app': 'func_name == "DLIOBenchmark.run"',
            'training': 'func_name == "DLIOBenchmark._train"',
            'compute': 'cat == "ai_framework"',
            'fetch_data': 'func_name.isin(["<module>.iter", "fetch-data.iter", "loop.iter"])',
            'data_loader': 'cat == "data_loader" & ~func_name.isin(["loop.iter", "loop.yield"])',
            'data_loader_fork': 'cat == "posix" & func_name == "fork"',
            'reader': 'cat == "reader"',
            # 'reader_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader")',
            'reader_posix_lustre': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader_lustre")',
            # 'reader_posix_ssd': 'cat.str.contains("posix|stdio") & cat.str.contains("_reader_ssd")',
            'checkpoint': 'cat == "checkpoint"',
            # 'checkpoint_posix': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint")',
            'checkpoint_posix_lustre': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint_lustre")',
            'checkpoint_posix_ssd': 'cat.str.contains("posix|stdio") & cat.str.contains("_checkpoint_ssd")',
            'other_posix': 'cat.isin(["posix", "stdio"])',
            # 'other_posix_lustre': 'cat.isin(["posix_lustre", "stdio_lustre"])',
            # 'other_posix_ssd': 'cat.isin(["posix_ssd", "stdio_ssd"])',
        }
    )
    layer_deps: Optional[Dict[str, Optional[str]]] = dc.field(
        default_factory=lambda: {
            'app': None,
            'training': 'app',
            'compute': 'training',
            'fetch_data': 'training',
            'data_loader': 'fetch_data',
            'data_loader_fork': 'fetch_data',
            'reader': 'data_loader',
            # 'reader_posix': 'reader',
            'reader_posix_lustre': 'reader',
            # 'reader_posix_ssd': 'reader_posix',
            'checkpoint': 'training',
            # 'checkpoint_posix': 'checkpoint',
            'checkpoint_posix_lustre': 'checkpoint',
            'checkpoint_posix_ssd': 'checkpoint',
            'other_posix': None,
            # 'other_posix_lustre': 'other_posix',
            # 'other_posix_ssd': 'other_posix',
        }
    )
    logical_views: Optional[Dict[str, Dict[str, Optional[str]]]] = dc.field(
        default_factory=lambda: {
            'file_name': {
                'file_dir': None,
                'file_pattern': None,
            },
            'proc_name': {
                'host_name': 'proc_name.str.split("#").str[1]',
                'proc_id': 'proc_name.str.split("#").str[2]',
                'thread_id': 'proc_name.str.split("#").str[3]',
            },
        }
    )
    name: str = "dlio"
    threaded_layers: Optional[List[str]] = dc.field(
        default_factory=lambda: [
            'data_loader',
            'data_loader_fork',
            'reader',
            # 'reader_posix',
            'reader_posix_lustre',
            # 'reader_posix_ssd',
        ]
    )
    unscored_metrics: Optional[List[str]] = dc.field(
        default_factory=lambda: [
            'consumer_rate',
            'producer_rate',
        ]
    )


@dc.dataclass
class AnalyzerConfig:
    checkpoint: Optional[bool] = True
    checkpoint_dir: Optional[str] = "${hydra:run.dir}/checkpoints"
    preset: Optional[AnalyzerPresetConfig] = MISSING
    quantile_stats: Optional[bool] = False
    time_approximate: Optional[bool] = True
    time_granularity: Optional[float] = MISSING
    time_resolution: Optional[float] = MISSING
    time_sliced: Optional[bool] = False


@dc.dataclass
class DarshanAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dfanalyzer.darshan.DarshanAnalyzer"
    time_granularity: Optional[float] = 1e3
    time_resolution: Optional[float] = 1e3


@dc.dataclass
class DFTracerAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dfanalyzer.dftracer.DFTracerAnalyzer"
    time_granularity: Optional[float] = 1e6
    time_resolution: Optional[float] = 1e6


@dc.dataclass
class RecorderAnalyzerConfig(AnalyzerConfig):
    _target_: str = "dfanalyzer.recorder.RecorderAnalyzer"
    time_granularity: Optional[float] = 1e7
    time_resolution: Optional[float] = 1e7


@dc.dataclass
class ClusterConfig:
    local_directory: Optional[str] = "/tmp/${hydra:job.name}-${oc.env:USER}/${oc.select:hydra.job.id,0}"


@dc.dataclass
class ExternalClusterConfig(ClusterConfig):
    _target_: str = "dfanalyzer.cluster.ExternalCluster"
    restart_on_connect: Optional[bool] = False
    scheduler_address: Optional[str] = MISSING


@dc.dataclass
class JobQueueClusterSchedulerConfig:
    dashboard_address: Optional[str] = None
    host: Optional[str] = dc.field(default_factory=socket.gethostname)


@dc.dataclass
class JobQueueClusterConfig(ClusterConfig):
    cores: int = 16  # ncores
    death_timeout: Optional[int] = 60
    job_directives_skip: Optional[List[str]] = dc.field(default_factory=list)
    job_extra_directives: Optional[List[str]] = dc.field(default_factory=list)
    log_directory: Optional[str] = ""
    memory: Optional[str] = None
    processes: Optional[int] = 1  # nnodes
    scheduler_options: Optional[JobQueueClusterSchedulerConfig] = dc.field(
        default_factory=JobQueueClusterSchedulerConfig
    )


@dc.dataclass
class LocalClusterConfig(ClusterConfig):
    _target_: str = "dask.distributed.LocalCluster"
    host: Optional[str] = None
    memory_limit: Optional[int] = None
    n_workers: Optional[int] = None
    processes: Optional[bool] = True
    silence_logs: Optional[int] = logging.CRITICAL


@dc.dataclass
class LSFClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.LSFCluster"
    use_stdin: Optional[bool] = True


@dc.dataclass
class PBSClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.PBSCluster"


@dc.dataclass
class SLURMClusterConfig(JobQueueClusterConfig):
    _target_: str = "dask_jobqueue.SLURMCluster"


@dc.dataclass
class OutputConfig:
    compact: Optional[bool] = False
    name: Optional[str] = ""
    root_only: Optional[bool] = True
    view_names: Optional[List[str]] = dc.field(default_factory=list)


@dc.dataclass
class ConsoleOutputConfig(OutputConfig):
    _target_: str = "dfanalyzer.output.ConsoleOutput"
    show_debug: Optional[bool] = False
    show_header: Optional[bool] = True


@dc.dataclass
class CSVOutputConfig(OutputConfig):
    _target_: str = "dfanalyzer.output.CSVOutput"


@dc.dataclass
class SQLiteOutputConfig(OutputConfig):
    _target_: str = "dfanalyzer.output.SQLiteOutput"
    run_db_path: Optional[str] = ""


@dc.dataclass
class CustomJobConfig(JobConf):
    name: str = "dfanalyzer"


@dc.dataclass
class CustomHelpConfig(HelpConf):
    app_name: str = "DFAnalyzer"
    header: str = "${hydra:help.app_name}: Data Flow Analyzer"
    footer: str = dc.field(
        default_factory=lambda: """
Powered by Hydra (https://hydra.cc)

Use --hydra-help to view Hydra specific help
    """.strip()
    )
    template: str = dc.field(
        default_factory=lambda: """
${hydra:help.header}

== Configuration groups ==

Compose your configuration from those groups (group=option)

$APP_CONFIG_GROUPS
== Config ==

Override anything in the config (foo.bar=value)

$CONFIG
${hydra:help.footer}
    """.strip()
    )


@dc.dataclass
class CustomLoggingConfig:
    version: int = 1
    formatters: Dict[str, Any] = dc.field(
        default_factory=lambda: {
            "simple": {
                "datefmt": "%H:%M:%S",
                "format": "[%(levelname)s] [%(asctime)s.%(msecs)03d] %(message)s [%(pathname)s:%(lineno)d]",
            }
        }
    )
    handlers: Dict[str, Any] = dc.field(
        default_factory=lambda: {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": "${hydra:runtime.output_dir}/${hydra:job.name}.log",
            },
        }
    )
    root: Dict[str, Any] = dc.field(
        default_factory=lambda: {
            "level": "INFO",
            "handlers": ["file"],
        }
    )
    disable_existing_loggers: bool = False


@dc.dataclass
class Config:
    defaults: List[Any] = dc.field(
        default_factory=lambda: [
            {"analyzer": "dftracer"},
            {"analyzer/preset": "posix"},
            {"hydra/job": "custom"},
            {"cluster": "local"},
            {"output": "console"},
            "_self_",
            {"override hydra/help": "custom"},
            {"override hydra/job_logging": "custom"},
        ]
    )
    analyzer: AnalyzerConfig = MISSING
    cluster: ClusterConfig = MISSING
    debug: Optional[bool] = False
    exclude_characteristics: Optional[List[str]] = dc.field(default_factory=list)
    logical_view_types: Optional[bool] = False
    metric_boundaries: Optional[ViewMetricBoundaries] = dc.field(default_factory=dict)
    output: OutputConfig = MISSING
    percentile: Optional[float] = None
    threshold: Optional[int] = None
    time_view_type: Optional[str] = COL_TIME_RANGE
    trace_path: str = MISSING
    verbose: Optional[bool] = False
    view_types: Optional[List[str]] = dc.field(default_factory=lambda: VIEW_TYPES)
    unoverlapped_posix_only: Optional[bool] = False


def init_hydra_config_store() -> ConfigStore:
    cs = ConfigStore.instance()
    cs.store(group="hydra/help", name="custom", node=dc.asdict(CustomHelpConfig()))
    cs.store(group="hydra/job", name="custom", node=CustomJobConfig)
    cs.store(group="hydra/job_logging", name="custom", node=CustomLoggingConfig)
    cs.store(name="config", node=Config)
    cs.store(group="analyzer", name="darshan", node=DarshanAnalyzerConfig)
    cs.store(group="analyzer", name="dftracer", node=DFTracerAnalyzerConfig)
    cs.store(group="analyzer", name="recorder", node=RecorderAnalyzerConfig)
    cs.store(group="analyzer/preset", name="posix", node=AnalyzerPresetConfigPOSIX)
    cs.store(group="analyzer/preset", name="dlio", node=AnalyzerPresetConfigDLIO)
    cs.store(group="cluster", name="external", node=ExternalClusterConfig)
    cs.store(group="cluster", name="local", node=LocalClusterConfig)
    cs.store(group="cluster", name="lsf", node=LSFClusterConfig)
    cs.store(group="cluster", name="pbs", node=PBSClusterConfig)
    cs.store(group="cluster", name="slurm", node=SLURMClusterConfig)
    cs.store(group="output", name="console", node=ConsoleOutputConfig)
    cs.store(group="output", name="csv", node=CSVOutputConfig)
    cs.store(group="output", name="sqlite", node=SQLiteOutputConfig)
    return cs
