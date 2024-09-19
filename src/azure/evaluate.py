def main():
    logging.basicConfig(filename='../pipeline.log', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--fcst", type=str, help="Path to forecast data")
    parser.add_argument("--test", type=str, help="Path to test data")
    # parser.add_argument("--train", type=str, help="Path to train data")
    parser.add_argument("--output", type=str, help="Path to output data")
    args = parser.parse_args()

    evaluate_forecasts(
        fcst_dir=args.fcst,
        test_file=args.test,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
