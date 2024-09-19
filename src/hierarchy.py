import logging

import numpy as np
import polars as pl
import scipy.sparse as sp_sparse
from itertools import product, chain

logger = logging.getLogger(__name__)


def get_hierarchy_groups(hierarchy_config):
    hierarchy_products = product(*hierarchy_config)
    hierarchy_chains = [tuple([elem]) for elem in chain(*hierarchy_config)]
    hierarchy_total = [tuple()]

    hierarchy_groups = list(
        chain(hierarchy_products, hierarchy_chains, hierarchy_total)
    )
    index = range(len(hierarchy_groups))
    hierarchy_groups = dict(zip(index, hierarchy_groups))
    return hierarchy_groups


def get_group_mappings(ids_df, group_key):
    if len(group_key) == 0:
        map_df = pl.DataFrame([[ids_df["unique_idx"]]]).rename({"column_0": "child_indexes"})
        return map_df

    map_df = (
        ids_df
        .group_by(group_key)
        .agg("unique_idx")
        .rename({"unique_idx": "child_indexes"})
        .sort(by=group_key)
    )

    return map_df


def get_group_aggregates(bts_df, ids_df, group_key):
    group_key = group_key + ("ds",)
    agg_df = (
        bts_df
        .join(ids_df, on="unique_id", how="left")
        .group_by(group_key)
        .agg(pl.col("y").sum())
        .sort(by=group_key)
    )
    return agg_df


def get_hierarchy_mappings(ids_df, hierarchy_config):
    hierarchy_groups = get_hierarchy_groups(hierarchy_config)

    id_cols = list(set(ids_df.columns).difference(["unique_id", "unique_idx"]))

    group_mappings_list = []
    for _, group_key in hierarchy_groups.items():
        group_mappings = get_group_mappings(ids_df, group_key)
        group_mappings_list.append(group_mappings)

    hts_mappings = (
        pl.concat(group_mappings_list, how="diagonal")
        .select(id_cols + ["child_indexes"])
        .fill_null("<...>")
        .with_columns(
            pl.col("child_indexes").list.len().alias("n_childs"),
            pl.concat_str(id_cols, separator="_").alias("unique_id")
        )
        .sort(by=["n_childs", "unique_id"], descending=True)
    )

    return hts_mappings


def get_hierarchy_aggregates(bts_df, ids_df, hierarchy_config):
    hierarchy_groups = get_hierarchy_groups(hierarchy_config)

    id_cols = list(set(ids_df.columns).difference(["unique_id", "unique_idx"]))

    group_df_list = []
    for _, group_key in hierarchy_groups.items():
        group_df = get_group_aggregates(bts_df, ids_df, group_key)
        group_df_list.append(group_df)

    hts_df = (
        pl.concat(group_df_list, how="diagonal")
        .select(id_cols + ["ds", "y"])
        .fill_null("<...>")
        .with_columns(
            pl.concat_str(id_cols, separator="_").alias("unique_id")
        )
        .sort(by=["unique_id", "ds"])
    )

    return hts_df


def get_S_arr(hts_mappings, sparse=True):
    M = len(hts_mappings)
    N = len(hts_mappings["child_indexes"][0])
    S_arr = np.zeros((M, N))

    index_rows = hts_mappings.select("child_indexes").iter_rows()
    for i, row in enumerate(index_rows):
        indexes = row[0]
        S_arr[i, indexes] = 1

    if sparse:
        S_arr = sp_sparse.csr_array(S_arr)

    return S_arr


def build_hierarchy(bts_df, ids_df, hierarchy_config):
    hts_df = get_hierarchy_aggregates(bts_df, ids_df, hierarchy_config)
    hts_mappings = get_hierarchy_mappings(ids_df, hierarchy_config)
    S_arr = get_S_arr(hts_mappings)
    
    hts_df = (
        hts_df
        .join(hts_mappings.select("unique_id", "n_childs"), on="unique_id", how="left")
        .sort(by=["n_childs", "unique_id", "ds"], descending=True)
        .drop("n_childs")
        .select(["unique_id", "ds", "y"])
    )

    hts_ids = hts_mappings.drop("child_indexes", "n_childs").unique(maintain_order=True)
    hts_ids = (
        hts_ids
        .with_columns(
            pl.Series(np.arange(len(hts_ids))).alias("unique_idx")
        )
    )

    return hts_df, hts_ids, S_arr
