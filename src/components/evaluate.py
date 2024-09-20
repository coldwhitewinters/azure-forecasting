import os
import logging
import argparse
from pathlib import Path

import polars as pl
from src.evaluation import calculate_metrics

logger = logging.getLogger(__name__)


def evaluate_component():
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fcst", type=str)
    parser.add_argument("--input-test", type=str)
    parser.add_argument("--output-metrics", type=str)
    args = parser.parse_args()

    logger.info("Loading forecast and test data")

    fcst_data = [pl.read_parquet(fcst_fp) for fcst_fp in Path(args.input_fcst).iterdir()]
    fcst_df = pl.concat(fcst_data)
    test_df = pl.read_parquet(args.input_test)

    logger.info("Calculating metrics")

    metrics_df = calculate_metrics(
        fcst_df=fcst_df,
        test_df=test_df
    )

    logger.info("Saving metrics")

    metrics_df.write_parquet(os.path.join(args.output_metrics, "metrics.parquet"))
