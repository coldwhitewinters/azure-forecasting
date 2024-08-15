import argparse
import os

from src.preprocessing import prepare_m5_data
from src.hierarchical import build_hierarchy
from src.forecast import forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/m5", help="path to input data")
    parser.add_argument("--output", type=str, default="output/", help="path to output data")
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="maximum number of timeseries to process"
    )
    args = parser.parse_args()

    processed_data_dir = os.path.join(args.input, "processed")

    prepare_m5_data(
        input_dir=args.input, 
        output_dir=processed_data_dir, 
        max_series=args.max_series
    )

    build_hierarchy(
        input_dir=processed_data_dir,
        output_dir=processed_data_dir
    )

    forecast(
        input_dir=processed_data_dir, 
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
