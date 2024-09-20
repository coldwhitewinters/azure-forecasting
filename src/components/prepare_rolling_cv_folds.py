import os
import logging
import argparse

import polars as pl
from src.evaluation import prepare_rolling_cv_folds

logger = logging.getLogger(__name__)


def prepare_rolling_cv_folds_component():
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output-train", type=str)
    parser.add_argument("--output-test", type=str)
    parser.add_argument("--freq", type=str)
    parser.add_argument(
        "--lags", type=str, nargs="+",
        help="Go back in time by this amount and start forecasting from there"
    )
    args = parser.parse_args()

    logger.info("Preparing rolling CV folds")

    ts_df = pl.read_parquet(args.input)

    cv_folds = prepare_rolling_cv_folds(
        ts_df=ts_df,
        freq=args.freq,
        lags=map(int, args.lags)
    )

    logger.info("Saving rolling CV folds")

    for lag, train_df in cv_folds:
        train_df.write_parquet(os.path.join(args.output_train, f"lag_{lag}.parquet"))

    logger.info("Saving test data")

    ts_df.write_parquet(args.output_test)


if __name__ == "__main__":
    prepare_rolling_cv_folds_component()
