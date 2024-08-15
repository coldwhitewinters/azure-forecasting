import polars as pl
import os
import json


def prepare_m5_data(input_dir, output_dir, max_series=None):
    sales_df = pl.read_csv(os.path.join(input_dir, "sales_train_evaluation.csv"))
    prices_df = pl.read_csv(os.path.join(input_dir, "sell_prices.csv"))
    calendar_df = pl.read_csv(os.path.join(input_dir, "calendar.csv"))

    ids_df = (
        sales_df
        .rename({"id": "unique_id"})
        .select(["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
        .with_columns(
            pl.col("unique_id").str.replace("_evaluation", "")
        )
        .sort(by="unique_id")
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

    if max_series:
        selected_series = ids_df.select("unique_id").unique().slice(0, max_series)
        ids_df = ids_df.filter(pl.col("unique_id").is_in(selected_series))
        data_df = data_df.filter(pl.col("unique_id").is_in(selected_series))

    os.makedirs(output_dir, exist_ok=True)
    data_df.write_parquet(os.path.join(output_dir, "data.parquet"))
    ids_df.write_parquet(os.path.join(output_dir, "data_ids.parquet"))
    with open(os.path.join(output_dir, "hierarchy_spec.json"), "w") as fp:
        json.dump(hierarchy_spec, fp)
