import os
import logging
import argparse
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def train_test_split(ts_df, freq, lag):
    start_date = ts_df["ds"].min()
    end_date = ts_df["ds"].max()
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    # dates = pl.date_range(start=start_date, end=end_date, interval=freq, eager=True)
    cutoff = dates[:-lag].max()
    ts_train_df = ts_df.filter(ts_df["ds"] <= cutoff)
    ts_test_df = ts_df.filter(ts_df["ds"] > cutoff)
    return ts_train_df, ts_test_df


def prepare_eval_data(input_dir, train_fp, test_fp, freq, lag):
    logger.info("Splitting data into train and test sets")

    ts_df = pl.read_parquet(os.path.join(input_dir, "hts.parquet"))

    if lag > 0:
        train_df, test_df = train_test_split(
            ts_df=ts_df,
            freq=freq,
            lag=lag
        )
    else:
        train_df = ts_df
        test_df = pl.DataFrame()

    logger.info("Saving train and test sets")
    train_df.write_parquet(train_fp)
    test_df.write_parquet(test_fp)


def main():
    logging.basicConfig(filename='../pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input data")
    parser.add_argument("--train", type=str, help="Path to output train data")
    parser.add_argument("--test", type=str, help="Path to output test data")
    parser.add_argument("--freq", type=str, help="Frequency of the data")
    parser.add_argument(
        "--lag", type=int,
        help="Go back in time by this amount and start forecasting from there"
    )
    args = parser.parse_args()

    prepare_eval_data(
        input_dir=args.input,
        train_fp=args.train,
        test_fp=args.test,
        freq=args.freq,
        lag=args.lag
    )


if __name__ == "__main__":
    main()
