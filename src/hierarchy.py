import os
import json
import argparse
import logging

import numpy as np
import polars as pl
import scipy.sparse as sp_sparse
from itertools import product, chain

logger = logging.getLogger(__name__)


def get_hierarchy_groups(hierarchy_spec):
    hierarchy_products = product(*hierarchy_spec)
    hierarchy_chains = [tuple([elem]) for elem in chain(*hierarchy_spec)]
    hierarchy_total = [tuple()]

    hierarchy_groups = list(
        chain(hierarchy_products, hierarchy_chains, hierarchy_total)
    )
    index = range(len(hierarchy_groups))
    hierarchy_groups = dict(zip(index, hierarchy_groups))
    return hierarchy_groups


def build_group(bts_df, ids_df, group_key):
    group_key = group_key + ("ds",)
    agg_df = (
        bts_df
        .join(ids_df, on="unique_id", how="left")
        .group_by(group_key)
        .agg(
            pl.col("y").sum(),
        )
        .sort(by=group_key)
    )
    return agg_df


def build_group_indexes(ids_df, group_key):
    index = np.arange(len(ids_df))

    if len(group_key) == 0:
        group_indexes_df = (
            pl.DataFrame([[pl.Series(index)]])
            .rename({"column_0": "index"})
        )
        return group_indexes_df

    group_indexes_df = (
        ids_df.with_columns(
            pl.Series(index).alias("index")
        )
        .group_by(group_key)
        .agg(pl.col("index"))
        .sort(by=group_key)
    )

    return group_indexes_df


def build_S_arr(hts_indexes, sparse=True):
    M = len(hts_indexes)
    N = len(hts_indexes["index"][0])
    S_arr = np.zeros((M, N))

    index_rows = hts_indexes.select("index").iter_rows()
    for i, row in enumerate(index_rows):
        indexes = row[0]
        S_arr[i, indexes] = 1

    if sparse:
        S_arr = sp_sparse.csr_array(S_arr)

    return S_arr


def build_hts(bts_df, ids_df, hierarchy_spec):
    hierarchy_groups = get_hierarchy_groups(hierarchy_spec)

    id_cols = ids_df.columns
    id_cols.remove("unique_id")

    group_indexes_list = []
    for _, group_key in hierarchy_groups.items():
        group_indexes = build_group_indexes(ids_df, group_key)
        group_indexes_list.append(group_indexes)

    hts_indexes = (
        pl.concat(group_indexes_list, how="diagonal")
        .select(id_cols + ["index"])
        .fill_null("<...>")
        .with_columns(
            pl.col("index").list.len().alias("size"),
            pl.concat_str(id_cols, separator="_").alias("unique_id")
        )
        .sort(by=["size", "unique_id"], descending=True)
    )

    S_arr = build_S_arr(hts_indexes)

    group_df_list = []
    for _, group_key in hierarchy_groups.items():
        group_df = build_group(bts_df, ids_df, group_key)
        group_df_list.append(group_df)

    hts_df = (
        pl.concat(group_df_list, how="diagonal")
        .select(id_cols + ["ds", "y"])
        .fill_null("<...>")
        .with_columns(
            pl.concat_str(id_cols, separator="_").alias("unique_id")
        )
        .join(hts_indexes.select("unique_id", "size"), on="unique_id", how="left")
        .sort(by=["size", "unique_id", "ds"], descending=True)
        .drop("size")
    )

    return hts_df, S_arr


def build_hierarchy(input_dir, output_dir):
    logger.info("Loading preprocessed data")

    data_df = pl.read_parquet(os.path.join(input_dir, "data.parquet"))
    ids_df = pl.read_parquet(os.path.join(input_dir, "data_ids.parquet"))
    with open(os.path.join(input_dir, "hierarchy_spec.json")) as fp:
        hierarchy_spec = json.load(fp)

    bts_df = (
        data_df
        .select("unique_id", "date", "dollar_sales")
        .rename({"date": "ds", "dollar_sales": "y"})
    )

    logger.info("Building time series hierarchy")

    hts_df, S_arr = build_hts(bts_df, ids_df, hierarchy_spec)

    id_cols = ids_df.columns
    hts_ids = hts_df.select(id_cols).unique(maintain_order=True)
    hts_df = hts_df.select(["unique_id", "ds", "y"])

    logger.info("Saving time series hierarchy")

    hts_ids.write_parquet(os.path.join(output_dir, "hts_ids.parquet"))
    hts_df.write_parquet(os.path.join(output_dir, "hts.parquet"))

    if isinstance(S_arr, np.ndarray):
        np.save(os.path.join(output_dir, "S_arr.npy"), S_arr)
    elif isinstance(S_arr, sp_sparse.csr_array) or isinstance(S_arr, sp_sparse.csc_array):
        sp_sparse.save_npz(os.path.join(output_dir, "S_arr.npy"), S_arr)
    else:
        raise ValueError("S_arr must be a numpy array or a sparse matrix")


if __name__ == "__main__":
    logging.basicConfig(filename='../pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input data")
    parser.add_argument("--output", type=str, help="Path to output data")
    args = parser.parse_args()

    build_hierarchy(
        input_dir=args.input,
        output_dir=args.output,
    )
