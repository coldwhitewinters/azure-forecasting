import polars as pl


def get_target_df(df, target_col, id_col="unique_id", date_col="ds"):
    target_df = (
        df
        .select(id_col, date_col, target_col)
        .rename({date_col: "ds", target_col: "y"})
    )
    return target_df


def get_regressors_df(df, regressor_cols, id_col="unique_id", date_col="ds"):
    regressors_df = (
        df
        .select(id_col, date_col, regressor_cols)
        .rename({date_col: "ds"})
    )
    return regressors_df
