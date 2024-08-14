from src.preprocessing import prepare_m5_data
from src.forecast import forecast
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to input data")
    parser.add_argument("--output", type=str, help="path to output data")
    parser.add_argument(
        "--max-series",
        type=int,
        help="maximum number of timeseries to process"
    )
    args = parser.parse_args()

    processed_data_dir = os.path.join(args.input, "processed")

    prepare_m5_data(
        input_dir=args.input, 
        output_dir=os.path.join(args.input, "processed"), 
        max_series=args.max_series
    )

    forecast(
        input_dir=processed_data_dir, 
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
