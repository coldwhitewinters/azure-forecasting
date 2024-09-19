#import sys
import json
import argparse
import logging
#from pathlib import Path

#ROOT_DIR = Path(__file__).absolute().parent.parent.parent
#sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.m5 import prepare_data

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output-data", type=str)
    parser.add_argument("--output-ids", type=str)
    parser.add_argument("--output-hierarchy-config", type=str)
    parser.add_argument("--max-series", type=int, default=None)
    args = parser.parse_args()

    logger.info("Starting data preparation")

    data_df, ids_df, hierarchy_config = prepare_data(
        input_dir=args.input,
        max_series=args.max_series
    )

    logger.info("Saving processed data")

    data_df.write_parquet(args.output_data)
    ids_df.write_parquet(args.output_ids)
    with open(args.output_hierarchy_config, "w") as fp:
        json.dump(hierarchy_config, fp)


if __name__ == "__main__":
    main()
