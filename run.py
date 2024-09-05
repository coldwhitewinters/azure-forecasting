import os
import argparse

from dask.distributed import Client

from src.preprocessing import prepare_m5_data
from src.hierarchy import build_hierarchy
from src.forecast import forecast


def main():
    client = Client(n_workers=16, threads_per_worker=1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/m5",
                        help="Path to input data")
    parser.add_argument("--output", type=str, default="output/",
                        help="Path to output data")
    parser.add_argument("--max-series", type=int, default=None,
                        help="Maximum number of time series to process")
    parser.add_argument("--horizon", type=int, help="Forecast horizon")
    parser.add_argument("--freq", type=str, help="Frequency of the data")
    parser.add_argument("--season-length", type=int, help="Seasonal length of the data")
    parser.add_argument("--model", type=str, help="Model to use for the forecasts")
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
        input_dir=args.input,
        output_dir=args.output,
        horizon=args.horizon,
        freq=args.freq,
        season_length=args.season_length,
        model=args.model,
        n_partitions=args.n_partitions
    )


if __name__ == "__main__":
    main()
