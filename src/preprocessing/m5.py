import os
import json
import logging
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


def prepare_data(input_dir, max_series=None):
    logger.info("Loading data")

    sales_df = pl.read_csv(os.path.join(input_dir, "sales_train_evaluation.csv"))
    prices_df = pl.read_csv(os.path.join(input_dir, "sell_prices.csv"))
    calendar_df = pl.read_csv(os.path.join(input_dir, "calendar.csv"))

    logger.info("Processing data")

    ids_df = (
        sales_df
        .rename({"id": "unique_id"})
        .select(["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
        .with_columns(
            pl.col("unique_id").str.replace("_evaluation", "")
        )
        .sort(by="unique_id")
    )

    ids_df = (
        ids_df
        .with_columns(
            pl.Series(np.arange(len(ids_df))).alias("unique_idx")
        )
        .select("unique_idx", *ids_df.columns)
    )

    calendar_df = (
        calendar_df
        .with_columns(
            pl.col("date").cast(pl.datatypes.Date),
            pl.col("d").str.replace("d_", "").cast(pl.datatypes.Int64)
        )
        .drop("weekday")
        .rename({"d": "i"})
    )

    data_df = (
        sales_df
        .rename({"id": "unique_id"})
        .with_columns(
            pl.col("unique_id").str.replace("_evaluation", "")
        )
        .drop(["item_id", "dept_id", "cat_id", "store_id", "state_id"])
        .unpivot(index=["unique_id"], variable_name="i", value_name="unit_sales")
        .with_columns(
            pl.col("i").str.replace("d_", "").cast(pl.datatypes.Int64)
        )
        .join(
            ids_df.select("unique_id", "item_id", "store_id"),
            on="unique_id",
            how="left"
        )
        .join(calendar_df, on="i", how="left")
        .join(prices_df, on=["item_id", "store_id", "wm_yr_wk"], how="left")
        .with_columns(
            (pl.col("unit_sales") * pl.col("sell_price")).alias("dollar_sales")
        )
        .drop("item_id", "store_id")
        .sort(by=["unique_id", "date"])
    )

    hierarchy_spec = [
        ["cat_id", "dept_id", "item_id"],
        ["state_id", "store_id"]
    ]

    logger.info("Finished data preparation")

    if max_series:
        logger.info(f"Keeping only {max_series} time series")
        selected_series = ids_df.select("unique_id").unique().slice(0, max_series)
        ids_df = ids_df.filter(pl.col("unique_id").is_in(selected_series))
        data_df = data_df.filter(pl.col("unique_id").is_in(selected_series))

    return data_df, ids_df, hierarchy_spec
