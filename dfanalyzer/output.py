import abc
import colorsys
import dask
import dataclasses as dc
import inflect
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Optional

from .constants import COL_FILE_NAME, COL_PROC_NAME, Layer, MiB
from .types import (
    AnalyzerResultType,
    RawStats,
    ViewKey,
    humanized_view_name,
)


@dc.dataclass
class OutputLayerMetrics:
    time: float
    count: int
    size: float
    ops: float
    bandwidth: float
    num_files: int
    num_processes: int
    u_time: Optional[float]
    u_count: Optional[int]
    u_size: Optional[float]


@dc.dataclass
class OutputSummary:
    job_time: float
    layer_metrics: Dict[Layer, OutputLayerMetrics]
    layers: List[Layer]
    time_granularity: float
    time_resolution: float
    total_count: int
    total_num_files: int
    total_num_nodes: int
    total_num_processes: int


class Output(abc.ABC):
    def __init__(
        self,
        compact: bool = False,
        name: str = "",
        root_only: bool = False,
        view_names: List[str] = [],
    ):
        self.compact = compact
        self.name = name
        # self.output_dir = HydraConfig.get().runtime.output_dir
        self.pluralize = inflect.engine()
        self.root_only = root_only
        self.view_names = view_names

    def handle_result(self, result: AnalyzerResultType):
        raise NotImplementedError

    def _create_summary(self, result: AnalyzerResultType, view_key: ViewKey) -> OutputSummary:
        flat_view = result.flat_views[view_key]
        raw_stats = dask.compute(result.raw_stats)[0]
        if isinstance(raw_stats, dict):
            raw_stats = RawStats(**raw_stats)
        summary = OutputSummary(
            job_time=float(raw_stats.job_time),
            layer_metrics={},
            layers=result.layers,
            time_granularity=float(raw_stats.time_granularity),
            time_resolution=float(raw_stats.time_resolution),
            total_count=0,
            total_num_files=0,
            total_num_nodes=0,
            total_num_processes=0,
        )
        is_process_based = view_key[-1] == COL_PROC_NAME
        time_metric = 'time_sum' if is_process_based else 'time_max'
        for layer in result.layers:
            if is_process_based:
                time = flat_view.get(f"{layer}_{time_metric}", pd.Series([0.0])).max()
            else:
                time = flat_view.get(f"{layer}_{time_metric}", pd.Series([0.0])).sum()
            count = flat_view.get(f"{layer}_count_sum", pd.Series([0])).sum()
            size = None
            if 'posix' in layer:
                size = flat_view.get(f"{layer}_size_sum", pd.Series([0.0])).sum()
            num_files = flat_view.get(f"{layer}_file_name_nunique", pd.Series([0.0])).max()
            num_processes = flat_view.get(f"{layer}_proc_name_nunique", pd.Series([0.0])).max()
            u_time_col = f"u_{layer}_{time_metric}"
            u_time = None
            u_count = None
            u_size = None
            if u_time_col in flat_view:
                u_time_mask = flat_view[u_time_col] > 0
                if is_process_based:
                    u_time = flat_view[u_time_col][u_time_mask].max()
                else:
                    u_time = flat_view[u_time_col][u_time_mask].sum()
                u_count = flat_view.get(f"{layer}_count_sum", pd.Series([0.0]))[u_time_mask].sum()
                if 'posix' in layer:
                    u_size = flat_view.get(f"{layer}_size_sum", pd.Series([0.0]))[u_time_mask].sum()
            summary.layer_metrics[layer] = OutputLayerMetrics(
                time=float(time),
                count=int(count),
                size=float(size) if size is not None else float('nan'),
                ops=float(count / time) if time > 0 else 0.0,
                bandwidth=float(size / time) if size and time else float('nan'),
                num_files=0 if pd.isna(num_files) else int(num_files),
                num_processes=0 if pd.isna(num_processes) else int(num_processes),
                u_time=float(u_time) if u_time is not None else None,
                u_count=int(u_count) if u_count is not None else None,
                u_size=float(u_size) if u_size is not None else None,
            )
            if not pd.isna(num_processes):
                summary.total_num_processes = max(summary.total_num_processes, int(num_processes))
            if not pd.isna(num_files):
                summary.total_num_files = summary.total_num_files + int(num_files)
            summary.total_count = summary.total_count + int(count)
        if (COL_FILE_NAME,) in result.flat_views:
            summary.total_num_files = len(result.flat_views[(COL_FILE_NAME,)])
        if (COL_PROC_NAME,) in result.flat_views:
            summary.total_num_processes = len(result.flat_views[(COL_PROC_NAME,)])
        return summary

    def _humanized_layer_name(self, name: str) -> str:
        return (
            name.replace('_', ' ')
            .title()
            .replace('Posix', 'POSIX')
            .replace('Stdio', 'STDIO')
            .replace('Lustre', '(Lustre)')
            .replace('Ssd', '(SSD)')
        )


class ConsoleOutput(Output):
    def __init__(
        self,
        compact: bool = False,
        name: str = "",
        root_only: bool = False,
        show_debug: bool = False,
        show_header: bool = True,
        view_names: List[str] = [],
    ):
        super().__init__(compact, name, root_only, view_names)
        self.show_debug = show_debug
        self.show_header = show_header

    def handle_result(self, result: AnalyzerResultType):
        print_objects = []
        for view_key in result.flat_views:
            summary = self._create_summary(result=result, view_key=view_key)
            summary_table = self._create_summary_table(summary=summary, view_key=view_key)
            layer_breakdown_table = self._create_layer_breakdown_table(summary=summary, view_key=view_key)
            print_objects.append(summary_table)
            print_objects.append(layer_breakdown_table)
        console = Console(record=True)
        console.print(*print_objects)

    def _create_layer_breakdown_table(self, summary: OutputSummary, view_key: ViewKey) -> Table:
        breakdown_table_title = "Layer Breakdown"
        show_overlap = False
        if len(summary.layers) > 1:
            breakdown_table_title += " (w/ overlap %)"
            show_overlap = True
        breakdown_table = Table(title=breakdown_table_title, title_style="bold cyan", expand=True)
        breakdown_table.add_column("Layer", style="bold")
        breakdown_table.add_column("Time (s)", justify="right")
        breakdown_table.add_column("Ops", justify="right")
        breakdown_table.add_column("Ops/sec", justify="right")
        breakdown_table.add_column("Size (MB)", justify="right")
        breakdown_table.add_column("Bandwidth (MB/s)", justify="right")
        for layer in summary.layers:
            layer_metrics = summary.layer_metrics[layer]
            if layer_metrics.count == 0:
                continue
            if show_overlap:
                time_str = self._format_val_with_ovlp_pct(layer_metrics.time, layer_metrics.u_time)
                count_str = self._format_val_with_ovlp_pct(layer_metrics.count, layer_metrics.u_count, fmt_int=True)
            else:
                time_str = self._format_val(layer_metrics.time)
                count_str = self._format_val(layer_metrics.count, fmt_int=True)
            size_str = '-'
            if not pd.isna(layer_metrics.size):
                size_value = layer_metrics.size / MiB
                u_size_value = layer_metrics.u_size / MiB if layer_metrics.u_size is not None else None
                if show_overlap:
                    size_str = self._format_val_with_ovlp_pct(size_value, u_size_value)
                else:
                    size_str = self._format_val(size_value)
            ops_str = f"{layer_metrics.ops:.3f}"
            bandwidth_str = '-'
            if not pd.isna(layer_metrics.bandwidth):
                bandwidth_str = f"{layer_metrics.bandwidth / MiB:.3f}"
            breakdown_table.add_row(
                self._humanized_layer_name(layer),
                time_str,
                count_str,
                ops_str,
                size_str,
                bandwidth_str,
            )
        return breakdown_table

    def _create_summary_table(self, summary: OutputSummary, view_key: ViewKey) -> Table:
        view_name = humanized_view_name(view_key, ' ')

        summary_table = Table(title=f"{view_name} Summary", title_style='bold green', expand=True)
        summary_table.add_column(header='Metric', style='bold')
        summary_table.add_column(header='Unit', style='italic')
        summary_table.add_column(header='Value', justify='right')

        summary_table.add_row('Job Time', 'seconds', f"{summary.job_time:.3f}")
        summary_table.add_row('Total Count', 'count', f"{summary.total_count:,}")
        summary_table.add_row('Total Files', 'count', f"{summary.total_num_files:,}")
        summary_table.add_row('Total Nodes', 'count', f"{summary.total_num_nodes:,}")
        summary_table.add_row('Total Processes', 'count', f"{summary.total_num_processes:,}")

        for layer in summary.layers:
            layer_name = self._humanized_layer_name(layer)
            layer_metrics = summary.layer_metrics[layer]
            if layer_metrics.count == 0:
                continue
            summary_table.add_row(f"{layer_name} Count", 'count', f"{layer_metrics.count:,}")
            if layer_metrics.size > 0:
                avg_xfer_size = layer_metrics.size / layer_metrics.count
                summary_table.add_row(f"{layer_name} Size", 'MB', f"{layer_metrics.size / MiB:.3f}")
                summary_table.add_row(f"{layer_name} Bandwidth", 'MB/s', f"{layer_metrics.bandwidth / MiB:.3f}")
                summary_table.add_row(f"{layer_name} Avg Transfer Size", 'MB', f"{avg_xfer_size / MiB:.3f}")

        return summary_table

    def _format_val(self, value: float, fmt_int=False) -> str:
        if value is None or value == 0:
            return '-'
        if fmt_int:
            return f"{int(value):,}"
        return f"{value:.3f}"

    def _format_val_with_ovlp_pct(self, value, u_value, fmt_int=False):
        value = value or 0
        if value == 0 or u_value is None:
            value = self._format_val(value, fmt_int)
            return f"{value} (" + ('-' * 4) + ")"
        u_value = u_value or 0
        ovlp_pct = max(0.0, 1.0 - (u_value / value))
        padded_percent = f"{int(round(ovlp_pct * 100)):>3d}%"
        color = self._percentage_color(ovlp_pct)
        value = self._format_val(value, fmt_int)
        return f"{value} ([{color}]{padded_percent}[/{color}])"

    def _percentage_color(self, percentage: float) -> str:
        """
        Convert overlap percentage (0.0-1.0) to color name in hex (for rich).
        0% = red, 100% = green
        """
        percentage = max(0.0, min(1.0, percentage))  # Clamp to [0, 1]
        # HSV: Hue 0.0 (red) to 0.33 (green)
        h = percentage * 0.33  # red to green
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


class CSVOutput(Output):
    def handle_result(self, result: AnalyzerResultType):
        raise NotImplementedError("CSVOutput is not implemented yet.")


class SQLiteOutput(Output):
    def __init__(
        self,
        compact: bool = False,
        group_behavior: bool = False,
        name: str = "",
        root_only: bool = False,
        run_db_path: str = "",
        view_names: List[str] = [],
    ):
        super().__init__(compact, name, root_only, view_names)
        self.run_db_path = run_db_path

    def handle_result(self, result: AnalyzerResultType):
        raise NotImplementedError("SQLiteOutput is not implemented yet.")
