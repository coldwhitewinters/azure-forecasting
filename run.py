import os
import argparse
import logging

from dask.distributed import Client

from src.preprocessing import prepare_m5_data
from src.hierarchy import build_hierarchy
from src.forecast import forecast
from src.split import prepare_eval_data
from src.metrics import evaluate_forecasts

logger = logging.getLogger(__name__)


def main():
    if os.path.exists('./pipeline.log'):
        os.remove('./pipeline.log')
    logging.basicConfig(filename='./pipeline.log', level=logging.INFO)

    client = Client(n_workers=16, threads_per_worker=1)  # noqa: F841

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/m5",
        help="Path to input data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Path to output data"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        help="Forecast horizon"
    )
    parser.add_argument(
        "--freq",
        type=str,
        help="Frequency of the data"
    )
    parser.add_argument(
        "--season-length",
        type=int,
        help="Seasonal length of the data"
    )
    parser.add_argument(
        "--lag",
        type=int,
        help="Go back in time by this amount and start forecasting from there"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for the forecasts"
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Maximum number of time series to process"
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=1,
        help="Number of partitions to use for the forecast"
    )
    args = parser.parse_args()

    prepare_m5_data(
        input_dir=args.input,
        output_dir=args.output,
        max_series=args.max_series
    )

    build_hierarchy(
        input_dir=args.output,
        output_dir=args.output
    )

    prepare_eval_data(
        input_dir=args.output,
        output_dir=args.output,
        freq=args.freq,
        lag=args.lag
    )

    forecast(
        input_dir=args.output,
        output_dir=args.output,
        horizon=args.horizon,
        freq=args.freq,
        season_length=args.season_length,
        model=args.model,
        n_partitions=args.n_partitions
    )

    evaluate_forecasts(
        input_dir=args.output,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
