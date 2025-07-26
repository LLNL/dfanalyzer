import dask.dataframe as dd
import itertools as it
import numpy as np
import pandas as pd
import portion as P


def nunique():
    return dd.Aggregation(
        name="nunique",
        chunk=lambda s: s.apply(lambda x: list(set(x))),
        agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
        finalize=lambda s1: s1.apply(lambda final: len(set(final))),
    )


def quantile_stats(min, max):
    def quantile_stats_finalize(values_list):
        if len(values_list) == 0:
            return [np.nan, np.nan, np.nan]
        values_array = np.array(values_list)
        q_min, q_max = np.quantile(values_array, [min, max])
        filtered_mask = (values_array >= q_min) & (values_array <= q_max)
        filtered_values = values_array[filtered_mask]
        if len(filtered_values) == 0:
            return [np.nan, np.nan, np.nan]
        return [np.mean(filtered_values), np.std(filtered_values), len(filtered_values)]

    return dd.Aggregation(
        f"q{min * 100:.0f}_q{max * 100:.0f}_stats",
        lambda s: s.apply(lambda x: x.replace(0, np.nan).dropna().tolist()),
        lambda s0: s0.obj.groupby(level=0).sum(),
        lambda s1: s1.apply(quantile_stats_finalize),
    )


def unique_set():
    return dd.Aggregation(
        'unique',
        lambda s: s.apply(lambda x: set() if pd.isna(x).any() else set(x)),
        lambda s0: s0.apply(lambda x: set(it.chain.from_iterable(x)) if not pd.isna(x).any() else set()),
        lambda s1: s1.apply(lambda x: set(x) if len(x) > 0 else pd.NA),
    )


def unique_set_flatten():
    def safe_union(x):
        # Handle null values
        if pd.isna(x).any() if hasattr(x, 'any') else pd.isna(x):
            return set()

        # If x is a Series, convert each element to set and union them
        if isinstance(x, pd.Series):
            result = set()
            for item in x:
                if pd.isna(item):
                    continue
                elif hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
                    try:
                        result.update(item)
                    except TypeError:
                        result.add(item)
                else:
                    result.add(item)
            return result

        # Handle scalar or single iterable
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            try:
                return set(x)
            except TypeError:
                return {x}
        else:
            return {x}

    def safe_agg_union(x):
        # Aggregate function to union all sets
        if isinstance(x, pd.Series):
            result = set()
            for item in x:
                if isinstance(item, set):
                    result.update(item)
                elif not pd.isna(item):
                    result.add(item)
            return result
        else:
            return {x} if not pd.isna(x) else set()

    return dd.Aggregation(
        'unique',
        lambda s: s.apply(safe_union),
        lambda s0: s0.agg(safe_agg_union),
        lambda s1: s1.apply(set),
    )


def union_portions():
    def union_s(s):
        emp = P.empty()
        for x in s:
            emp = emp | x
        return emp

    def fin(s):
        val = 0.0
        for i in s:
            if not i.is_empty():
                val += i.upper - i.lower
        return val

    return dd.Aggregation(
        'portion',
        union_s,
        union_s,
        fin,
    )
