import os
import logging

import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)
os.environ["NIXTLA_ID_AS_COL"] = "1"


def get_rolling_cv_fold(ts_df, freq, lag):
    if lag == 0:
        train_df = ts_df
        test_df = pl.DataFrame()
        return train_df, test_df

    start_date = ts_df["ds"].min()
    end_date = ts_df["ds"].max()
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    # dates = pl.date_range(start=start_date, end=end_date, interval=freq, eager=True)
    cutoff = dates[:-lag].max()
    train_df = ts_df.filter(ts_df["ds"] <= cutoff)
    return train_df


def prepare_rolling_cv_folds(ts_df, freq, lags):
    cv_folds = ((lag, get_rolling_cv_fold(ts_df, freq, lag)) for lag in lags)
    return cv_folds


def calculate_metrics(fcst_df, test_df, eps=0):
    eval_df = fcst_df.join(test_df, on=["unique_id", "ds"], how="inner")
    train_df = test_df.filter(pl.col("ds") < fcst_df["ds"].min())

    insample_naive_mae = (
        train_df
        .group_by("unique_id")
        .agg(
            pl.col("y").diff().abs().mean().alias("insample_naive_mae"),
        )
    )

    err_expr = (pl.col("y") - pl.col("y_hat")).alias("err")
    mse_expr = (pl.col("err")**2).mean().alias("mse")
    mae_expr = pl.col("err").abs().mean().alias("mae")
    rmse_expr = (pl.col("err")**2).mean().sqrt().alias("rmse")
    mape_expr = (pl.col("err").abs() / pl.max_horizontal(pl.col("y").abs(), eps)).mean().alias("mape")
    smape_expr = (pl.col("err").abs() / pl.max_horizontal(pl.col("y").abs() + pl.col("y_hat").abs(), eps)).mean().alias("smape")
    wape_expr = (pl.col("err").abs().sum() / pl.col("y").abs().sum()).alias("wape")
    mase_expr = (pl.col("mae") / pl.col("insample_naive_mae")).alias("mase")
    rmsse_expr = (pl.col("rmse") / pl.col("insample_naive_mae")).alias("rmsse")

    metrics_df = (
        eval_df
        .with_columns(err_expr)
        .group_by(["cutoff", "unique_id"])
        .agg(
            mse_expr,
            mae_expr,
            rmse_expr,
            mape_expr,
            smape_expr,
            wape_expr,
        )
        .join(
            insample_naive_mae,
            on="unique_id",
            how="left",
        )
        .with_columns(
            mase_expr,
            rmsse_expr
        )
        .drop("insample_naive_mae")
    )

    return metrics_df
