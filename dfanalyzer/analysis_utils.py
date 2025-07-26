import hashlib
import numpy as np
import os
import pandas as pd
import re
from typing import Union

from .constants import (
    COL_COUNT,
    COL_FILE_NAME,
    COL_PROC_NAME,
    COL_SIZE,
    COL_TIME,
    COL_TIME_RANGE,
    COL_TIME_START,
    FILE_PATTERN_PLACEHOLDER,
    PROC_NAME_SEPARATOR,
    SIZE_BINS,
    SIZE_BIN_SUFFIXES,
)


def fix_dtypes(df: pd.DataFrame):
    int_cols = []
    int_cols.extend([col for col in df.columns if '_bin_' in col])
    int_cols.extend([col for col in df.columns if col.endswith('_count')])
    int_cols.extend([col for col in df.columns if col.endswith('_nunique')])
    double_cols = []
    double_cols.extend([col for col in df.columns if col.endswith('_bw')])
    double_cols.extend([col for col in df.columns if col.endswith('_intensity')])
    double_cols.extend([col for col in df.columns if col.endswith('_ops')])
    double_cols.extend([col for col in df.columns if col.endswith('_time')])
    double_cols.extend([col for col in df.columns if col.endswith('_slope')])
    double_cols.extend([col for col in df.columns if col.endswith('_pct')])
    size_cols = [col for col in df.columns if col.endswith('_size')]
    # df[int_cols] = df[int_cols].astype('int64[pyarrow]')
    # df[double_cols] = df[double_cols].astype('double[pyarrow]')
    # df[size_cols] = df[size_cols].astype('int64[pyarrow]')
    df[int_cols] = df[int_cols].astype('Int32')
    df[double_cols] = df[double_cols].astype('Float64')
    df[size_cols] = df[size_cols].astype('Int64')
    return df


def fix_size_values(df: pd.DataFrame):
    size_cols = [col for col in df.columns if 'size' in col]
    df[size_cols] = df[size_cols].replace(0, np.nan)
    return df


def set_app_name(df: pd.DataFrame):
    return df.assign(
        app_name=lambda df: df.index.get_level_values(COL_PROC_NAME)
        .str.split(PROC_NAME_SEPARATOR)
        .str[0]
        .astype("string[pyarrow]"),
    )


def set_host_name(df: pd.DataFrame):
    return df.assign(
        app_name=lambda df: df.index.get_level_values(COL_PROC_NAME)
        .str.split(PROC_NAME_SEPARATOR)
        .str[1]
        .astype("string[pyarrow]"),
    )


def set_file_dir(df: pd.DataFrame):
    if COL_FILE_NAME not in df.index.names:
        return df
    return df.assign(
        file_dir=df.index.get_level_values(COL_FILE_NAME).map(os.path.dirname).astype("string[pyarrow]"),
    )


def set_file_pattern(df: pd.DataFrame):
    if COL_FILE_NAME not in df.index.names:
        return df

    def _apply_regex(file_name: str):
        return re.sub('[0-9]+', FILE_PATTERN_PLACEHOLDER, file_name)

    return df.assign(
        file_pattern=df.index.get_level_values(COL_FILE_NAME).map(_apply_regex).astype("string[pyarrow]"),
    )


def set_id(ix: Union[tuple, str, int]):
    ix_str = '_'.join(map(str, ix)) if isinstance(ix, tuple) else str(ix)
    return int(hashlib.md5(ix_str.encode()).hexdigest(), 16)


def set_proc_name_parts(df: pd.DataFrame):
    if COL_PROC_NAME not in df.index.names:
        return df

    proc_names = df.index.get_level_values(COL_PROC_NAME)

    first_proc_name_parts = proc_names[0].split(PROC_NAME_SEPARATOR)

    if first_proc_name_parts[0] == 'app':
        return df.assign(
            proc_name_parts=proc_names.str.split(PROC_NAME_SEPARATOR),
            app_name=lambda df: df.proc_name_parts.str[0].astype(str),
            host_name=lambda df: df.proc_name_parts.str[1].astype(str),
            # node_name=lambda df: pd.NA,
            proc_id=lambda df: df.proc_name_parts.str[2].astype(str),
            rank=lambda df: pd.NA,
            thread_id=lambda df: df.proc_name_parts.str[3].astype(str),
        ).drop(columns=['proc_name_parts'])

    return df.assign(
        proc_name_parts=proc_names.str.split(PROC_NAME_SEPARATOR),
        app_name=lambda df: df.proc_name_parts.str[0].astype(str),
        host_name=lambda df: df.proc_name_parts.str[1].astype(str),
        # node_name=lambda df: df.proc_name_parts.str[1].astype(str),
        proc_id=lambda df: pd.NA,
        rank=lambda df: df.proc_name_parts.str[2].astype(str),
        thread_id=lambda df: pd.NA,
    ).drop(columns=['proc_name_parts'])


def set_size_bins(df: pd.DataFrame):
    df['size_bin_temp'] = pd.cut(
        df['size'],
        bins=SIZE_BINS,
        labels=SIZE_BIN_SUFFIXES,
        right=True,
        include_lowest=True,
    )
    size_bin_dummies = pd.get_dummies(df['size_bin_temp'], prefix='size_bin', dtype=int)
    df = pd.concat([df, size_bin_dummies], axis=1)
    df = df.drop('size_bin_temp', axis=1)
    return df


def set_unique_counts(df: pd.DataFrame, layer: str):
    unique_cols = [col for col in df.columns if col.endswith('_unique')]
    for unique_col in unique_cols:
        if COL_FILE_NAME in unique_col and 'posix' not in layer:
            continue
        nunique_col = unique_col.replace('_unique', '_nunique')
        # Handle null values before applying len() - null values should map to 0
        df[nunique_col] = df[unique_col].map(lambda x: len(x) if pd.notna(x) else 0).astype('uint64[pyarrow]')
    return df.drop(columns=unique_cols)


def split_duration_records_vectorized(
    df: pd.DataFrame,
    time_granularity: float,
    time_resolution: float,
) -> pd.DataFrame:
    # Convert duration column to numpy array
    durations = df[COL_TIME].to_numpy()

    # Calculate number of chunks needed for each row
    n_chunks = np.ceil(durations / time_granularity).astype(int)
    max_chunks = n_chunks.max()

    if max_chunks == 0:
        df[COL_TIME_RANGE] = df[COL_TIME_START] // (time_granularity * time_resolution)
        df[COL_TIME_RANGE] = df[COL_TIME_RANGE].astype('uint64[pyarrow]')
        return df.copy()

    # Create expansion indices
    row_idx = np.arange(len(df))
    repeated_idx = np.repeat(row_idx, n_chunks)

    # Create the expanded dataframe
    result_df = df.iloc[repeated_idx].copy()

    # Calculate chunk numbers for each expanded row
    chunk_numbers = np.concatenate([np.arange(n) for n in n_chunks])

    # Calculate time values
    is_last_chunk = chunk_numbers == (n_chunks.repeat(n_chunks) - 1)
    time_values = np.full(len(chunk_numbers), time_granularity)

    # Handle remainders for last chunks
    remainders = durations % time_granularity
    remainders = np.where(remainders == 0, time_granularity, remainders)
    time_values[is_last_chunk] = remainders.repeat(n_chunks)[is_last_chunk]

    # Update time column
    result_df[COL_TIME] = time_values

    # Calculate and update timestamps
    ts_base = df[COL_TIME_START].to_numpy()
    ts_offsets = chunk_numbers * time_granularity * time_resolution  # Convert to microseconds
    result_df[COL_TIME_START] = ts_base.repeat(n_chunks) + ts_offsets

    result_df[COL_TIME_RANGE] = result_df[COL_TIME_START] // (time_granularity * time_resolution)
    result_df[COL_TIME_RANGE] = result_df[COL_TIME_RANGE].astype('uint64[pyarrow]')

    counts = df[COL_COUNT].to_numpy()
    expanded_counts = counts.repeat(n_chunks)
    # Divide counts by number of chunks for each original row
    result_df[COL_COUNT] = expanded_counts / n_chunks.repeat(n_chunks)

    sizes = df[COL_SIZE].to_numpy()
    expanded_sizes = sizes.repeat(n_chunks)
    # Divide sizes by number of chunks for each original row
    result_df[COL_SIZE] = expanded_sizes / n_chunks.repeat(n_chunks)

    return result_df.reset_index(drop=True)
