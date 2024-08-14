import polars as pl
import os


def prepare_m5_data(input_dir, output_dir, max_series=None):
    sales_df = pl.read_csv(os.path.join(input_dir, "sales_train_evaluation.csv"))
    prices_df = pl.read_csv(os.path.join(input_dir, "sell_prices.csv"))
    calendar_df = pl.read_csv(os.path.join(input_dir, "calendar.csv"))

    hierarchy_df = (
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

    sales_df = (
        sales_df
        .rename({"id": "unique_id"})
        .with_columns(
            pl.col("unique_id").str.replace("_evaluation", "")
        )
        .drop(["item_id", "dept_id", "cat_id", "store_id", "state_id"])
        .unpivot(index=["unique_id"], variable_name="i", value_name="y")
        .with_columns(
            pl.col("i").str.replace("d_", "").cast(pl.datatypes.Int64)
        )
        .join(
            hierarchy_df.select("unique_id", "item_id", "store_id"),
            on="unique_id",
            how="left"
        )
        .join(calendar_df, on="i", how="left")
        .join(prices_df, on=["item_id", "store_id", "wm_yr_wk"], how="left")
        .drop("item_id", "store_id")
        .sort(by=["unique_id", "date"])
    )

    if max_series:
        selected_series = hierarchy_df.select("unique_id").unique().slice(0, max_series)
        hierarchy_df = hierarchy_df.filter(pl.col("unique_id").is_in(selected_series))
        sales_df = sales_df.filter(pl.col("unique_id").is_in(selected_series))

    os.makedirs(output_dir, exist_ok=True)
    sales_df.write_parquet(os.path.join(output_dir, "sales_df.parquet"))
    hierarchy_df.write_parquet(os.path.join(output_dir, "hierarchy_df.parquet"))
