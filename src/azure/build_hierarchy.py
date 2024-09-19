import os
import json
import argparse
import logging

import polars as pl
import numpy as np
import scipy.sparse as sp_sparse

from src.hierarchy import build_hierarchy

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(filename="pipeline.log", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bts", type=str)
    parser.add_argument("--input-ids", type=str)
    parser.add_argument("--input-hierarchy-config", type=str)
    parser.add_argument("--output-hts", type=str)
    parser.add_argument("--output-ids", type=str)
    parser.add_argument("--output-smatrix", type=str)
    args = parser.parse_args()

    ids_df = pl.read_parquet(args.input_ids)
    y_df = pl.read_parquet(args.input_bts)

    with open(args.input_hierarchy_config, "r") as fp:
        hierarchy_config = json.load(fp)

    logger.info("Building time series hierarchy")

    hts_df, hts_ids, S_arr = build_hierarchy(y_df, ids_df, hierarchy_config)

    logger.info("Saving time series hierarchy")

    hts_ids.write_parquet(args.output_ids)
    hts_df.write_parquet(args.output_hts)

    if isinstance(S_arr, np.ndarray):
        np.save(os.path.join(args.output_smatrix, "S_arr.npy"), S_arr)
    elif isinstance(S_arr, sp_sparse.csr_array) or isinstance(S_arr, sp_sparse.csc_array):
        sp_sparse.save_npz(os.path.join(args.output_smatrix, "S_arr.npz"), S_arr)
    else:
        raise ValueError("S_arr must be a numpy array or a sparse matrix")


if __name__ == "__main__":
    main()
