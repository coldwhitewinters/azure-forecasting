import os
import argparse
import logging
import json

import numpy as np
import scipy.sparse as sp_sparse
from dask.distributed import Client

from src.preprocessing.m5 import prepare_data
from src.preprocessing.utils import get_target_df
from src.hierarchy import build_hierarchy

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
        default="output/default",
        help="Path to output data"
    )
    # parser.add_argument(
    #     "--horizon",
    #     type=int,
    #     help="Forecast horizon"
    # )
    # parser.add_argument(
    #     "--freq",
    #     type=str,
    #     help="Frequency of the data"
    # )
    # parser.add_argument(
    #     "--season-length",
    #     type=int,
    #     help="Seasonal length of the data"
    # )
    # parser.add_argument(
    #     "--lag",
    #     type=int,
    #     help="Go back in time by this amount and start forecasting from there"
    # )
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     help="Model to use for the forecasts"
    # )
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

    logger.info("Starting data preparation")

    data_df, ids_df, hierarchy_spec = prepare_data(
        input_dir=args.input,
        max_series=args.max_series
    )

    logger.info("Saving processed data")

    os.makedirs(args.output, exist_ok=True)
    data_df.write_parquet(os.path.join(args.output, "data.parquet"))
    ids_df.write_parquet(os.path.join(args.output, "data_ids.parquet"))
    with open(os.path.join(args.output, "hierarchy_spec.json"), "w") as fp:
        json.dump(hierarchy_spec, fp)

    logger.info("Retrieving target and regressors")

    y_df = get_target_df(data_df, target_col="unit_sales", date_col="date")

    logger.info("Building time series hierarchy")

    hts_df, hts_ids, S_arr = build_hierarchy(y_df, ids_df, hierarchy_spec)

    logger.info("Saving time series hierarchy")

    hts_ids.write_parquet(os.path.join(args.output, "hts_ids.parquet"))
    hts_df.write_parquet(os.path.join(args.output, "hts.parquet"))

    if isinstance(S_arr, np.ndarray):
        np.save(os.path.join(args.output, "S_arr.npy"), S_arr)
    elif isinstance(S_arr, sp_sparse.csr_array) or isinstance(S_arr, sp_sparse.csc_array):
        sp_sparse.save_npz(os.path.join(args.output, "S_arr.npz"), S_arr)
    else:
        raise ValueError("S_arr must be a numpy array or a sparse matrix")


if __name__ == "__main__":
    main()
