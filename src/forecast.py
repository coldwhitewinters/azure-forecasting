import os
import argparse
import logging

import polars as pl
import pandas as pd
from dask.distributed import Client
import dask.dataframe as dd
import dask_mpi
from statsforecast import StatsForecast
import statsforecast.models

logger = logging.getLogger(__name__)
os.environ["NIXTLA_ID_AS_COL"] = "1"


def forecast(
    input_dir,
    output_dir,
    horizon=1,
    freq="M",
    season_length=1,
    model=statsforecast.models.Naive,
    n_partitions=1
):
    n_cpus = os.cpu_count()
    logger.info(f"We have {n_cpus} cores available")

    logger.info("Loading timeseries hierarchy")
    hts_ddf = dd.read_parquet(os.path.join(input_dir, "hts.parquet"))
    # ids_df = pl.read_parquet(os.path.join(input_dir, "hts_ids.parquet"))

    logger.info("Starting forecast")

    def forecast_func(df, horizon, freq, model):
        df["ds"] = pd.to_datetime(df["ds"])

        forecaster = StatsForecast(
            models=[model],
            freq=freq,
            n_jobs=1
        )
        fcst = forecaster.forecast(df=df, h=horizon)
        fcst = fcst.rename(columns={model.alias: "y_hat"})
        return fcst

    model_cls = getattr(statsforecast.models, model)
    meta = hts_ddf.head(0).rename(columns={"y": "y_hat"})
    meta["ds"] = pd.to_datetime(meta["ds"])
    fcst_df = (
        hts_ddf
        .repartition(npartitions=n_partitions)
        .map_partitions(
            forecast_func,
            horizon=horizon,
            freq=freq,
            model=model_cls(season_length=season_length),
            meta=meta
        )
        .compute()
    )
    fcst_df = pl.DataFrame(fcst_df)

    logger.info("Finished forecast")

    logger.info("Writing forecast")
    os.makedirs(output_dir, exist_ok=True)
    fcst_df.write_parquet(os.path.join(output_dir, "fcst.parquet"))


if __name__ == "__main__":
    logging.basicConfig(filename='../pipeline.log', level=logging.INFO)

    dask_mpi.initialize()
    client = Client()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to input data")
    parser.add_argument("--output", type=str, help="path to output data")
    parser.add_argument("--horizon", type=int, help="forecast horizon")
    parser.add_argument("--freq", type=str, help="frequency of the data")
    parser.add_argument("--season-length", type=int, help="seasonal length of the data")
    parser.add_argument("--model", type=str, help="model to use for the forecasts")
    parser.add_argument(
        "--n-partitions",
        type=int,
        help="number of partitions to use for distributing input dataframe"
    )
    args = parser.parse_args()

    forecast(
        input_dir=args.input,
        output_dir=args.output,
        horizon=args.horizon,
        freq=args.freq,
        season_length=args.season_length,
        model=args.model,
        n_partitions=args.n_partitions
    )
