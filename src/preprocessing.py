import polars as pl
from pathlib import Path
from datetime import datetime

def prepare_m5_data(data_dir):
    data_dir = Path(data_dir)
    
    sales_df = pl.read_csv(data_dir / "sales_train_evaluation.csv")
    prices_df = pl.read_csv(data_dir / "sell_prices.csv")
    calendar_df = pl.read_csv(data_dir / "calendar.csv")

    hierarchy_df = (
        sales_df
        .rename({"id": "unique_id"})
        .select(["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
        .with_columns(
            pl.col("unique_id").str.replace("_evaluation", "")
        )
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
        .join(hierarchy_df.select("unique_id", "item_id", "store_id"), on="unique_id", how="left")
        .join(calendar_df, on="i", how="left")
        .join(prices_df, on=["item_id", "store_id", "wm_yr_wk"], how="left")
        .drop("item_id", "store_id")
    )

    processed_data_dir = data_dir / "processed"
    processed_data_dir.mkdir(exist_ok=True)

    sales_df.write_parquet(processed_data_dir / "sales_df.parquet")
    hierarchy_df.write_parquet(processed_data_dir / "hierarchy_df.parquet")
