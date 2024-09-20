import argparse
import logging

import polars as pl
from src.preprocessing.utils import get_target_df

logger = logging.getLogger(__name__)


def prepare_target_and_regressors_component():
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output-target", type=str)
    parser.add_argument("--id-col", type=str, default="unique_id")
    parser.add_argument("--date-col", type=str, default="ds")
    parser.add_argument("--target-col", type=str, default="y")
    args = parser.parse_args()

    data_df = pl.read_parquet(args.input)

    logger.info("Retrieving target and regressors")

    y_df = get_target_df(
        data_df,
        target_col=args.target_col,
        date_col=args.date_col,
        id_col=args.id_col
    )

    logger.info("Saving target")

    y_df.write_parquet(args.output_target)


if __name__ == "__main__":
    prepare_target_and_regressors_component()
