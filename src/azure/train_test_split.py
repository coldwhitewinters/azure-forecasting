import logging
import argparse

from src.evaluation import prepare_eval_data

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(filename='pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output-train", type=str)
    parser.add_argument("--output-test", type=str)
    parser.add_argument("--freq", type=str)
    parser.add_argument(
        "--lags", type=str, nargs="+",
        help="Go back in time by this amount and start forecasting from there"
    )
    args = parser.parse_args()

    prepare_eval_data(
        input_fp=args.input,
        train_dir=args.output_train,
        test_fp=args.output_test,
        freq=args.freq,
        lags=map(int, args.lags)
    )


if __name__ == "__main__":
    main()
