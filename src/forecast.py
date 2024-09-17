import os
import argparse
import logging
from pathlib import Path

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
    ts_fp,
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

    hts_df = pl.read_parquet(ts_fp)
    hts_dd = dd.read_parquet(ts_fp)

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
    meta = hts_dd.head(0).rename(columns={"y": "y_hat"})
    meta["ds"] = pd.to_datetime(meta["ds"])
    fcst_df = (
        hts_dd
        .repartition(npartitions=n_partitions)
        .groupby("unique_id")
        .apply(
            forecast_func,
            horizon=horizon,
            freq=freq,
            model=model_cls(season_length=season_length),
            meta=meta
        )
        .compute()
    )
    fcst_df = fcst_df.reset_index(drop=True)
    fcst_df = pl.DataFrame(fcst_df).with_columns(pl.col("ds").cast(hts_df["ds"].dtype))
    fcst_df = fcst_df.with_columns(cutoff=fcst_df["ds"].min())

    logger.info("Finished forecast")

    logger.info("Writing forecast")
    os.makedirs(output_dir, exist_ok=True)
    lag = ts_fp.stem.split("_")[-1]
    fcst_df.write_parquet(os.path.join(output_dir, f"fcst_{lag}.parquet"))


def forecast_folds(
    input_dir,
    output_dir,
    horizon=1,
    freq="M",
    season_length=1,
    model=statsforecast.models.Naive,
    n_partitions=1
):
    input_dir = Path(input_dir)

    for ts_fp in input_dir.iterdir():
        forecast(
            ts_fp=ts_fp,
            output_dir=output_dir,
            horizon=horizon,
            freq=freq,
            season_length=season_length,
            model=model,
            n_partitions=n_partitions
        )


def main():
    logging.basicConfig(filename='../pipeline.log', level=logging.INFO)

    dask_mpi.initialize()
    client = Client()  # noqa F841

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input data")
    parser.add_argument("--output", type=str, help="Path to output data")
    parser.add_argument("--horizon", type=int, help="Forecast horizon")
    parser.add_argument("--freq", type=str, help="Frequency of the data")
    parser.add_argument("--season-length", type=int, help="Seasonal length of the data")
    parser.add_argument("--model", type=str, help="Model to use for the forecasts")
    parser.add_argument(
        "--n-partitions",
        type=int,
        help="Number of partitions to use for distributing input dataframe"
    )
    args = parser.parse_args()

    forecast_folds(
        input_dir=args.input,
        output_dir=args.output,
        horizon=args.horizon,
        freq=args.freq,
        season_length=args.season_length,
        model=args.model,
        n_partitions=args.n_partitions
    )


if __name__ == "__main__":
    main()
