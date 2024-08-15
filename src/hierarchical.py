import polars as pl
from itertools import product, chain
import os
import json


def get_hierarchy_groups(hierarchy_spec):
    hierarchy_products = product(*hierarchy_spec)
    hierarchy_chains = [tuple([elem]) for elem in chain(*hierarchy_spec)]
    hierarchy_total = [tuple()]

    levels = chain(hierarchy_products, hierarchy_chains, hierarchy_total)
    return list(levels)


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


def build_hts(bts_df, ids_df, hierarchy_spec):
    hierarchy_groups = get_hierarchy_groups(hierarchy_spec)

    group_df_list = []
    for group_key in hierarchy_groups:
        group_df = build_group(bts_df, ids_df, group_key)
        group_df_list.append(group_df)

    id_cols = ids_df.columns
    id_cols.remove("unique_id")
    all_cols = id_cols + ["ds", "y"]

    hts_df = (
        pl.concat(group_df_list, how="diagonal")
        .select(*all_cols)
        .fill_null("<AGG>")
        .with_columns(
            pl.concat_str(id_cols, separator="_").alias("unique_id")
        )
        .sort(by=["unique_id", "ds"])
    )

    return hts_df


def build_hierarchy(input_dir, output_dir):
    data_df = pl.read_parquet(os.path.join(input_dir, "data.parquet"))
    ids_df = pl.read_parquet(os.path.join(input_dir, "data_ids.parquet"))
    with open(os.path.join(input_dir, "hierarchy_spec.json")) as fp:
        hierarchy_spec = json.load(fp)

    bts_df = (
        data_df
        .select("unique_id", "date", "dollar_sales")
        .rename({"date": "ds", "dollar_sales": "y"})
    )

    hts_df = build_hts(bts_df, ids_df, hierarchy_spec)

    id_cols = ids_df.columns
    hts_ids = hts_df.select(*id_cols).unique()
    hts_df = hts_df.select(["unique_id", "ds", "y"])

    hts_ids.write_parquet(os.path.join(output_dir, "hts_ids.parquet"))
    hts_df.write_parquet(os.path.join(output_dir, "hts.parquet"))
