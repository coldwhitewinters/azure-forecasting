import os
import argparse
import logging
from pathlib import Path

import polars as pl
import dask.dataframe as dd
import dask_mpi
from dask.distributed import Client
from src.forecast import forecast

logger = logging.getLogger(__name__)


def forecast_component():
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)

    dask_mpi.initialize()
    client = Client()  # noqa F841

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output-fcst", type=str)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--freq", type=str)
    parser.add_argument("--season-length", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--n-partitions", type=int)
    args = parser.parse_args()

    logger.info("Loading timeseries")

    input_dir = Path(args.input)

    logger.info("Starting forecast")

    for ts_fp in input_dir.iterdir():
        ts_dd = dd.read_parquet(ts_fp, split_row_groups=True)
        output_schema = pl.read_parquet(ts_fp).schema

        fcst_df = forecast(
            ts_dd=ts_dd,
            output_schema=output_schema,
            horizon=args.horizon,
            freq=args.freq,
            season_length=args.season_length,
            model=args.model,
            n_partitions=args.n_partitions
        )

        logger.info("Writing forecast")

        lag = ts_fp.stem.split("_")[-1]
        fcst_df.write_parquet(os.path.join(args.output_fcst, f"fcst_{lag}.parquet"))

    logger.info("Finished forecast")


if __name__ == "__main__":
    forecast_component()
