import os
import logging

import polars as pl
import pandas as pd
from statsforecast import StatsForecast
import statsforecast.models

logger = logging.getLogger(__name__)
os.environ["NIXTLA_ID_AS_COL"] = "1"


def statsforecaster(df, horizon, freq, model):
    df["ds"] = pd.to_datetime(df["ds"])

    forecaster = StatsForecast(
        models=[model],
        freq=freq,
        n_jobs=1
    )
    fcst = forecaster.forecast(df=df, h=horizon)
    fcst = fcst.rename(columns={model.alias: "y_hat"})
    return fcst


def forecast(
    ts_dd,
    output_schema,
    horizon=1,
    freq="M",
    season_length=1,
    model=statsforecast.models.Naive,
    n_partitions=1
):
    n_cpus = os.cpu_count()
    logger.info(f"We have {n_cpus} cores available")

    model_cls = getattr(statsforecast.models, model)
    meta = ts_dd.head(0).rename(columns={"y": "y_hat"})
    meta["ds"] = pd.to_datetime(meta["ds"])
    fcst_df = (
        ts_dd
        .repartition(npartitions=n_partitions)
        .groupby("unique_id")
        .apply(
            statsforecaster,
            horizon=horizon,
            freq=freq,
            model=model_cls(season_length=season_length),
            meta=meta
        )
        .compute()
    )
    fcst_df = fcst_df.reset_index(drop=True)
    fcst_df = pl.DataFrame(fcst_df).cast(dtypes=output_schema)
    fcst_df = fcst_df.with_columns(cutoff=fcst_df["ds"].min())

    return fcst_df
