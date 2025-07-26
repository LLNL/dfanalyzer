import dask.dataframe as dd
import darshan as d
import glob
import numpy as np
import os
import pandas as pd

from .analyzer import Analyzer
from .constants import COL_TIME_END, COL_TIME_START, IOCategory
from .types import RawStats

TRACE_COL_MAPPING = {
    'end_time': COL_TIME_END,
    'start_time': COL_TIME_START,
}


class DarshanAnalyzer(Analyzer):
    job_time: float = 0.0

    def analyze_trace(
        self,
        trace_path,
        view_types,
        exclude_characteristics=...,
        extra_columns=None,
        extra_columns_fn=None,
        logical_view_types=False,
        metric_boundaries=...,
        percentile=None,
        threshold=None,
        time_view_type=None,
        unoverlapped_posix_only=False,
    ):
        if not trace_path.endswith('.darshan') and not os.path.isdir(trace_path):
            raise ValueError(f"Invalid trace path: {trace_path}. Must be a directory or a .darshan file.")

        reports = []
        if os.path.isdir(trace_path):
            trace_path = os.path.join(trace_path, '*.darshan')
            trace_paths = glob.glob(trace_path)
            for trace_path in trace_paths:
                report = d.DarshanReport(trace_path, read_all=True)
                reports.append(report)
        else:
            report = d.DarshanReport(trace_path, read_all=True)
            reports.append(report)

        self.reports = reports
        self.job_time = max(map(self._calculate_job_time, reports))

        if all('DXT_POSIX' in report.records for report in reports):
            # Let the analyzer do read_trace etc as normal
            return super().analyze_trace(
                trace_path=trace_path,
                view_types=view_types,
                exclude_characteristics=exclude_characteristics,
                extra_columns=extra_columns,
                extra_columns_fn=extra_columns_fn,
                logical_view_types=logical_view_types,
                metric_boundaries=metric_boundaries,
                percentile=percentile,
                threshold=threshold,
                time_view_type=time_view_type,
                unoverlapped_posix_only=unoverlapped_posix_only,
            )

        if any(view_type not in ['file_name', 'proc_name'] for view_type in view_types):
            raise ValueError("Only 'file_name' and 'proc_name' view types are supported for non-DXT traces.")

        is_slope_based = threshold is not None

        file_name_df = pd.concat(map(self._create_file_name_view, reports), ignore_index=True)
        file_name_ddf = (
            dd.from_pandas(file_name_df, npartitions=len(reports))
            .persist()
            .repartition(partition_size='256MB')
            .persist()
        )
        file_name_view = file_name_ddf.groupby(['file_name', 'proc_name']).sum().persist()

        raw_stats = RawStats(
            job_time=self.job_time,
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
            total_count=len(file_name_view),
        )

        # return file_name_view
        return self._analyze_main_view(
            main_view=file_name_view,
            metrics=metrics,
            percentile=percentile,
            threshold=threshold,
            view_types=view_types,
            is_slope_based=is_slope_based,
            raw_stats=raw_stats,
            exclude_characteristics=exclude_characteristics,
        )

    def read_trace(self, trace_path, extra_columns, extra_columns_fn):
        df = pd.concat(map(self._create_dxt_dataframe, self.reports), ignore_index=True)
        return dd.from_pandas(df, npartitions=len(self.reports))

    def compute_job_time(self, traces: dd.DataFrame) -> float:
        return self.job_time

    def _calculate_job_time(self, report: d.DarshanReport) -> float:
        job = report.metadata['job']
        if 'start_time' in job:
            job_time = job['end_time'] - job['start_time']
        else:
            job_time = job['end_time_sec'] - job['start_time_sec']
        return job_time

    def _create_dxt_dataframe(self, report: d.DarshanReport) -> pd.DataFrame:
        # Get the DXT_POSIX records
        dxt_df = pd.DataFrame(report.records['DXT_POSIX'].to_df())
        # Initialize data structures
        dxt_rows = []
        # Process each record
        for _, record in dxt_df.iterrows():
            file_id = record['id']
            rank = record['rank']
            host_name = record['hostname']
            file_name = report.data['name_records'][file_id]
            proc_name = f"app#localhost#{rank}#0"

            # Process read segments
            if not record['read_segments'].empty:
                read_segments = record['read_segments']

                # Convert dataframe to dict of lists for faster processing
                lengths = read_segments['length'].tolist()
                start_times = read_segments['start_time'].tolist()
                end_times = read_segments['end_time'].tolist()
                offsets = read_segments['offset'].tolist()

                # Create a batch of rows
                for i in range(len(lengths)):
                    dxt_rows.append(
                        {
                            'file_name': file_name,
                            'proc_name': proc_name,
                            'size': lengths[i],
                            'end_time': end_times[i],
                            'start_time': start_times[i],
                            'func_name': 'read',
                            'host_name': host_name,
                            'io_cat': IOCategory.READ.value,
                            'time_range': int(start_times[i] * self.time_granularity),
                            'cat': 'posix',
                            'acc_pat': 0,  # Would need more logic for random access patterns
                            'count': 1,
                            'time': end_times[i] - start_times[i],
                        }
                    )

            # Process write segments
            if not record['write_segments'].empty:
                write_segments = record['write_segments']

                # Convert dataframe to dict of lists for faster processing
                lengths = write_segments['length'].tolist()
                start_times = write_segments['start_time'].tolist()
                end_times = write_segments['end_time'].tolist()
                offsets = write_segments['offset'].tolist()

                # Create a batch of rows
                for i in range(len(lengths)):
                    dxt_rows.append(
                        {
                            'file_name': file_name,
                            'proc_name': proc_name,
                            'size': lengths[i],
                            'end_time': end_times[i],
                            'start_time': start_times[i],
                            'func_name': 'write',
                            'host_name': host_name,
                            'io_cat': IOCategory.WRITE.value,
                            'time_range': int(start_times[i] * self.time_granularity),
                            'cat': 'posix',
                            'acc_pat': 0,  # Would need more logic for random access patterns
                            'count': 1,
                            'time': end_times[i] - start_times[i],
                        }
                    )

        # Create the final dataframe in one go
        return pd.DataFrame(dxt_rows).rename(columns=TRACE_COL_MAPPING)

    def _create_file_name_view(self, report: d.DarshanReport) -> pd.DataFrame:
        posix_df = report.records['POSIX'].to_df()
        file_name_df = pd.DataFrame.from_dict(report.name_records, orient='index', columns=['file_name'])
        file_name_view = (
            posix_df['counters']
            .set_index(['rank', 'id'])
            .merge(
                posix_df['fcounters'].set_index(['rank', 'id']),
                left_index=True,
                right_index=True,
            )
        )
        if 'STDIO' in report.records:
            stdio_df = report.records['STDIO'].to_df()
            file_name_view = file_name_view.merge(
                stdio_df['counters'].set_index(['rank', 'id']),
                left_index=True,
                right_index=True,
                how='outer',
            ).merge(
                stdio_df['fcounters'].set_index(['rank', 'id']),
                left_index=True,
                right_index=True,
                how='outer',
            )
        file_name_view = (
            file_name_view.merge(
                file_name_df,
                left_on='id',
                right_index=True,
            )
            .reset_index()
            .assign(proc_name=lambda x: 'app#localhost#' + x['rank'].astype(str) + '#0')
            # .set_index(['proc_name', 'file_name'])
            .drop(columns=['id', 'rank'])
            .query('~(file_name.str.startswith("<") and file_name.str.endswith(">"))')
        )
        file_name_view['time'] = (
            file_name_view['POSIX_F_READ_TIME'].fillna(0)
            + file_name_view['POSIX_F_WRITE_TIME'].fillna(0)
            + file_name_view['POSIX_F_META_TIME'].fillna(0)
            + file_name_view['STDIO_F_READ_TIME'].fillna(0)
            + file_name_view['STDIO_F_WRITE_TIME'].fillna(0)
            + file_name_view['STDIO_F_META_TIME'].fillna(0)
        )
        file_name_view['read_time'] = file_name_view['POSIX_F_READ_TIME'].fillna(0) + file_name_view[
            'STDIO_F_READ_TIME'
        ].fillna(0)
        file_name_view['write_time'] = file_name_view['POSIX_F_WRITE_TIME'].fillna(0) + file_name_view[
            'STDIO_F_WRITE_TIME'
        ].fillna(0)
        file_name_view['metadata_time'] = file_name_view['POSIX_F_META_TIME'].fillna(0) + file_name_view[
            'STDIO_F_META_TIME'
        ].fillna(0)
        file_name_view['data_time'] = file_name_view['read_time'] + file_name_view['write_time']
        file_name_view['close_time'] = np.nan
        file_name_view['open_time'] = np.nan
        file_name_view['seek_time'] = np.nan
        file_name_view['stat_time'] = np.nan
        file_name_view['count'] = (
            file_name_view['POSIX_OPENS'].fillna(0)
            + file_name_view['POSIX_FILENOS'].fillna(0)
            + file_name_view['POSIX_DUPS'].fillna(0)
            + file_name_view['POSIX_READS'].fillna(0)
            + file_name_view['POSIX_WRITES'].fillna(0)
            + file_name_view['POSIX_SEEKS'].fillna(0)
            + file_name_view['POSIX_STATS'].fillna(0)
            # + file_name_view['POSIX_MMAPS'].fillna(0)
            + file_name_view['POSIX_FSYNCS'].fillna(0)
            + file_name_view['POSIX_FDSYNCS'].fillna(0)
            + file_name_view['STDIO_OPENS'].fillna(0)
            + file_name_view['STDIO_FDOPENS'].fillna(0)
            + file_name_view['STDIO_READS'].fillna(0)
            + file_name_view['STDIO_WRITES'].fillna(0)
            + file_name_view['STDIO_SEEKS'].fillna(0)
            + file_name_view['STDIO_FLUSHES'].fillna(0)
        )
        file_name_view['read_count'] = file_name_view['POSIX_READS'].fillna(0) + file_name_view['STDIO_READS'].fillna(
            0
        )
        file_name_view['write_count'] = file_name_view['POSIX_WRITES'].fillna(0) + file_name_view[
            'STDIO_WRITES'
        ].fillna(0)
        file_name_view['data_count'] = file_name_view['read_count'] + file_name_view['write_count']
        file_name_view['metadata_count'] = (
            file_name_view['POSIX_OPENS'].fillna(0)
            + file_name_view['POSIX_FILENOS'].fillna(0)
            + file_name_view['POSIX_DUPS'].fillna(0)
            + file_name_view['POSIX_SEEKS'].fillna(0)
            + file_name_view['POSIX_STATS'].fillna(0)
            # + file_name_view['POSIX_MMAPS'].fillna(0)
            + file_name_view['POSIX_FSYNCS'].fillna(0)
            + file_name_view['POSIX_FDSYNCS'].fillna(0)
            + file_name_view['STDIO_OPENS'].fillna(0)
            + file_name_view['STDIO_FDOPENS'].fillna(0)
            + file_name_view['STDIO_SEEKS'].fillna(0)
            + file_name_view['STDIO_FLUSHES'].fillna(0)
        )
        file_name_view['close_count'] = np.nan
        file_name_view['open_count'] = file_name_view['POSIX_OPENS'].fillna(0) + file_name_view['STDIO_OPENS'].fillna(
            0
        )
        file_name_view['seek_count'] = file_name_view['POSIX_SEEKS'].fillna(0) + file_name_view['STDIO_SEEKS'].fillna(
            0
        )
        file_name_view['stat_count'] = (file_name_view['POSIX_STATS'].fillna(0)).astype(int)
        file_name_view['size_min'] = np.nan
        file_name_view['size_max'] = np.maximum(
            file_name_view['POSIX_MAX_BYTE_READ'].fillna(0),
            file_name_view['POSIX_MAX_BYTE_WRITTEN'].fillna(0),
        )
        file_name_view['read_min'] = np.nan
        file_name_view['read_max'] = np.maximum(
            file_name_view['POSIX_MAX_BYTE_READ'].fillna(0),
            file_name_view['STDIO_MAX_BYTE_READ'].fillna(0),
        )
        file_name_view['write_min'] = np.nan
        file_name_view['write_max'] = np.maximum(
            file_name_view['POSIX_MAX_BYTE_WRITTEN'].fillna(0),
            file_name_view['STDIO_MAX_BYTE_WRITTEN'].fillna(0),
        )
        file_name_view['size'] = (
            file_name_view['POSIX_BYTES_READ'].fillna(0)
            + file_name_view['POSIX_BYTES_WRITTEN'].fillna(0)
            + file_name_view['STDIO_BYTES_READ'].fillna(0)
            + file_name_view['STDIO_BYTES_WRITTEN'].fillna(0)
        )
        file_name_view['read_size'] = file_name_view['POSIX_BYTES_READ'].fillna(0) + file_name_view[
            'STDIO_BYTES_READ'
        ].fillna(0)
        file_name_view['write_size'] = file_name_view['POSIX_BYTES_WRITTEN'].fillna(0) + file_name_view[
            'STDIO_BYTES_WRITTEN'
        ].fillna(0)
        file_name_view['data_size'] = file_name_view['read_size'] + file_name_view['write_size']
        file_name_view['sequential_time'] = np.nan
        file_name_view['sequential_count'] = file_name_view['POSIX_SEQ_READS'].fillna(0) + file_name_view[
            'POSIX_SEQ_WRITES'
        ].fillna(0)
        file_name_view['sequential_size'] = np.nan
        file_name_view['random_time'] = np.nan
        file_name_view['random_count'] = file_name_view['count'] - file_name_view['sequential_count']
        file_name_view['random_size'] = np.nan
        posix_cols = [col for col in file_name_view.columns if col.startswith('POSIX_')]
        stdio_cols = [col for col in file_name_view.columns if col.startswith('STDIO_')]
        file_name_view = file_name_view.drop(columns=posix_cols + stdio_cols)
        count_cols = [col for col in file_name_view.columns if col.endswith('_count')]
        size_cols = [col for col in file_name_view.columns if col.endswith('_size')]
        time_cols = [col for col in file_name_view.columns if col.endswith('_time')]
        file_name_view[count_cols] = file_name_view[count_cols].astype('Int64')
        file_name_view[size_cols] = file_name_view[size_cols].astype('Int64')
        file_name_view[time_cols] = file_name_view[time_cols].astype('Float64')
        return file_name_view
