import dask.dataframe as dd
import json
import numpy as np
import pandas as pd
from dask.distributed import Future, get_client
from typing import List, Union

from .analyzer import Analyzer
from .constants import COL_FUNC_NAME, COL_TIME, COL_TIME_END, COL_TIME_START, IO_CATS
from .types import ViewType


CAT_POSIX = 0
CAT_MAPPING = {
    0: 'posix',
}
DROPPED_COLS = [
    'app',
    'bandwidth',
    'file_id',
    'hostname',
    'index',
    'level',
    'proc',
    'proc_id',
    'rank',
    'tend',
    'thread_id',
    'tmid',
    'tstart',
]
TRACE_COL_MAPPING = {
    'duration': COL_TIME,
    'func_id': COL_FUNC_NAME,
    'tend': COL_TIME_END,
    'tstart': COL_TIME_START,
}


class RecorderAnalyzer(Analyzer):
    def read_trace(self, trace_path, extra_columns, extra_columns_fn):
        self.global_min_max = self._load_global_min_max(trace_path=trace_path)
        traces = dd.read_parquet(trace_path).rename(columns=TRACE_COL_MAPPING)
        return traces

    def postread_trace(
        self,
        traces: dd.DataFrame,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
        traces[COL_TIME] = traces[COL_TIME].astype('Float64')
        traces['acc_pat'] = traces['acc_pat'].astype('Int8')
        traces['count'] = 1
        traces['count'] = traces['count'].astype('Int64')
        traces['io_cat'] = traces['io_cat'].astype('Int8')
        time_ranges = self._compute_time_ranges(
            global_min_max=self.global_min_max,
            time_granularity=self.time_granularity,
        )
        traces = (
            traces[(traces['cat'] == CAT_POSIX) & (traces['io_cat'].isin(IO_CATS))]
            .map_partitions(self._set_time_ranges, time_ranges=time_ranges)
            .drop(columns=DROPPED_COLS, errors='ignore')
        )
        traces['cat'] = 'posix'
        traces['cat'] = traces['cat'].astype('string')
        return traces

    def compute_total_count(self, traces: dd.DataFrame) -> int:
        return traces[(traces['cat'] == CAT_POSIX) & (traces['io_cat'].isin(IO_CATS))].reduction(len, sum).persist()

    @staticmethod
    def _compute_time_ranges(global_min_max: dict, time_granularity: int):
        tmid_min, tmid_max = global_min_max['tmid']
        time_ranges = np.arange(tmid_min, tmid_max, time_granularity)
        return get_client().scatter(time_ranges)

    @staticmethod
    def _load_global_min_max(trace_path: str) -> dict:
        with open(f"{trace_path}/global.json") as file:
            global_min_max = json.load(file)
        return global_min_max

    @staticmethod
    def _set_time_ranges(df: pd.DataFrame, time_ranges: Union[Future, np.ndarray]):
        if isinstance(time_ranges, Future):
            time_ranges = time_ranges.result()
        return df.assign(time_range=np.digitize(df['tmid'], bins=time_ranges, right=True))
