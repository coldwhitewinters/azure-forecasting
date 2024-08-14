import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive
import os


def forecast(input_dir, output_dir):
    sales_df = pl.read_parquet(os.path.join(input_dir, "sales_df.parquet"))
    hierarchy_df = pl.read_parquet(os.path.join(input_dir, "hierarchy_df.parquet"))

    y_df = (
        sales_df
        .select("unique_id", "date", "y")
        .rename({"date": "ds"})
    )
    #unique_ids = y_df.select("unique_id").unique()

    forecaster = StatsForecast(
        models=[AutoETS(season_length=1)],
        freq="1d",
        n_jobs=-1,
        fallback_model=Naive()
    )
    fcst_df = forecaster.forecast(h=28, df=y_df)
    fcst_df = fcst_df.rename({"AutoETS": "y"})

    os.makedirs(output_dir, exist_ok=True)
    fcst_df.write_parquet(os.path.join(output_dir, "fcst.parquet"))
