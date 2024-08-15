import polars as pl
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive
import os


def forecast(input_dir, output_dir):
    hts_df = pl.read_parquet(os.path.join(input_dir, "hts.parquet"))
    #ids_df = pl.read_parquet(os.path.join(input_dir, "hts_ids.parquet"))

    forecaster = StatsForecast(
        models=[AutoETS(season_length=7)],
        freq="1d",
        n_jobs=-1,
        fallback_model=Naive()
    )
    fcst_df = forecaster.forecast(h=28, df=hts_df)
    fcst_df = fcst_df.rename({"AutoETS": "y"})

    os.makedirs(output_dir, exist_ok=True)
    fcst_df.write_parquet(os.path.join(output_dir, "fcst.parquet"))
