import dask
import warnings
from dataclasses import dataclass
from distributed import Client
from hydra import compose, initialize
from hydra.core.hydra_config import DictConfig, HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import List, Union, Optional

from .analyzer import Analyzer
from .cluster import ClusterType, ExternalCluster
from .config import init_hydra_config_store
from .dftracer import DFTracerAnalyzer
from .output import ConsoleOutput, CSVOutput, SQLiteOutput
from .recorder import RecorderAnalyzer
from .types import ViewType

try:
    from .darshan import DarshanAnalyzer
except ModuleNotFoundError:
    DarshanAnalyzer = Analyzer

AnalyzerType = Union[DarshanAnalyzer, DFTracerAnalyzer, RecorderAnalyzer]
OutputType = Union[ConsoleOutput, CSVOutput, SQLiteOutput]

# Suppress Dask warnings that are not relevant to the user
dask.config.set({"dataframe.query-planning-warning": False})

# Suppress FutureWarnings related to pandas grouper
warnings.filterwarnings(
    action="ignore",
    message=".*grouper",
    category=FutureWarning,
)


@dataclass
class DFAnalyzerInstance:
    analyzer: Analyzer
    client: Client
    cluster: ClusterType
    hydra_config: DictConfig
    output: OutputType

    def analyze_trace(
        self,
        percentile: Optional[float] = None,
        view_types: Optional[List[ViewType]] = None,
    ):
        """Analyze the trace using the configured analyzer."""
        return self.analyzer.analyze_trace(
            exclude_characteristics=self.hydra_config.exclude_characteristics,
            logical_view_types=self.hydra_config.logical_view_types,
            metric_boundaries=OmegaConf.to_object(self.hydra_config.metric_boundaries),
            percentile=self.hydra_config.percentile if not percentile else percentile,
            time_view_type=self.hydra_config.time_view_type,
            trace_path=self.hydra_config.trace_path,
            unoverlapped_posix_only=self.hydra_config.unoverlapped_posix_only,
            view_types=self.hydra_config.view_types if not view_types else view_types,
        )

    def shutdown(self):
        """Shutdown the Dask client and cluster."""
        self.client.close()
        if hasattr(self.cluster, 'close'):
            self.cluster.close()


def init_with_hydra(hydra_overrides: List[str]):
    with initialize(version_base=None, config_path=None):
        init_hydra_config_store()
        hydra_config = compose(
            config_name="config",
            overrides=hydra_overrides,
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(hydra_config)
    cluster = instantiate(hydra_config.cluster)
    if isinstance(cluster, ExternalCluster):
        client = Client(cluster.scheduler_address)
    else:
        client = Client(cluster)
    analyzer = instantiate(
        hydra_config.analyzer,
        debug=hydra_config.debug,
        verbose=hydra_config.verbose,
    )
    output = instantiate(hydra_config.output)
    return DFAnalyzerInstance(
        analyzer=analyzer,
        client=client,
        cluster=cluster,
        hydra_config=hydra_config,
        output=output,
    )
