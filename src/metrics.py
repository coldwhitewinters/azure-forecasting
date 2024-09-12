import os
import argparse
import logging

import polars as pl

logger = logging.getLogger(__name__)
os.environ["NIXTLA_ID_AS_COL"] = "1"


def calculate_metrics(fcst_df, test_df, train_df, eps=0):
    eval_df = fcst_df.join(test_df, on=["unique_id", "ds"], how="inner")

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
        .group_by("unique_id")
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


def evaluate_forecasts(input_dir, output_dir):
    logger.info("Loading forecast and test data")

    fcst_df = pl.read_parquet(os.path.join(input_dir, "fcst.parquet"))
    hts_train_df = pl.read_parquet(os.path.join(input_dir, "hts_train.parquet"))
    hts_test_df = pl.read_parquet(os.path.join(input_dir, "hts_test.parquet"))

    logger.info("Calculating metrics")

    metrics_df = calculate_metrics(
        fcst_df=fcst_df,
        train_df=hts_train_df,
        test_df=hts_test_df
    )

    overall_metrics = (
        pl.concat([
            metrics_df.drop("unique_id").median(),
            metrics_df.drop("unique_id").mean(),
        ]).with_columns(
            agg=pl.Series(["median", "mean"]),
        )
    )

    logger.info("Saving metrics")

    metrics_df.write_parquet(os.path.join(output_dir, "metrics.parquet"))
    overall_metrics.write_csv(os.path.join(output_dir, "overall_metrics.csv"))


def main():
    logging.basicConfig(filename='../pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input data")
    parser.add_argument("--output", type=str, help="Path to output data")
    args = parser.parse_args()

    evaluate_forecasts(input_dir=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
