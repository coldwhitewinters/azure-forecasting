import sys
import argparse
import logging
from pathlib import Path
import polars as pl

#ROOT_DIR = Path(__file__).absolute().parent.parent.parent
#sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.utils import get_target_df

logger = logging.getLogger(__name__)


def main():
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
    main()
