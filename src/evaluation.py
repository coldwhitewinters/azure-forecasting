import os
import argparse
import logging
from pathlib import Path
import polars as pl

logger = logging.getLogger(__name__)
os.environ["NIXTLA_ID_AS_COL"] = "1"


def calculate_metrics(fcst_df, test_df, eps=0):
    eval_df = fcst_df.join(test_df, on=["unique_id", "ds"], how="inner")

    #insample_naive_mae = (
    #    train_df
    #    .group_by("unique_id")
    #    .agg(
    #        pl.col("y").diff().abs().mean().alias("insample_naive_mae"),
    #    )
    #)

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
        #.join(
        #    insample_naive_mae,
        #    on="unique_id",
        #    how="left",
        #)
        #.with_columns(
        #    mase_expr,
        #    rmsse_expr
        #)
        #.drop("insample_naive_mae")
    )

    return metrics_df


def evaluate_forecasts(fcst_dir, test_file, output_dir):
    logger.info("Loading forecast and test data")

    fcst_data = [pl.read_parquet(fcst_fp) for fcst_fp in Path(fcst_dir).iterdir()]
    #train_data = [pl.read_parquet(train_fp) for train_fp in Path(train_dir).iterdir()]
    fcst_df = pl.concat(fcst_data)
    test_df = pl.read_parquet(test_file)

    logger.info("Calculating metrics")

    metrics_df = calculate_metrics(
        fcst_df=fcst_df,
        test_df=test_df
    )

    # lag_metrics = (
    #     metrics_df
    #     .drop("cutoff")
    #     .group_by("unique_id")
    #     .agg(
    #         pl.col("*").median(),
    #         pl.col("*").mean()
    #     )
    # )

    # overall_metrics = (
    #     pl.concat([
    #         lag_metrics.drop("unique_id", "mean").median(),
    #         lag_metrics.drop("unique_id", "median").mean(),
    #     ]).with_columns(
    #         pl.Series(["median", "mean"]).alias("agg"),
    #     )
    # )

    logger.info("Saving metrics")

    metrics_df.write_parquet(os.path.join(output_dir, "metrics.parquet"))
    #lag_metrics.write_parquet(os.path.join(output_dir, "lag_metrics.parquet"))
    #overall_metrics.write_csv(os.path.join(output_dir, "overall_metrics.csv"))


def main():
    logging.basicConfig(filename='../pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--fcst", type=str, help="Path to forecast data")
    parser.add_argument("--test", type=str, help="Path to test data")
    # parser.add_argument("--train", type=str, help="Path to train data")
    parser.add_argument("--output", type=str, help="Path to output data")
    args = parser.parse_args()

    evaluate_forecasts(
        fcst_dir=args.fcst,
        test_file=args.test,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
