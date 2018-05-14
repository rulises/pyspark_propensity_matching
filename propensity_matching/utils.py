"""Utility functions that may be useful at any stage."""

from typing import List, Tuple, Optional
from math import floor

import pyspark.sql as pys
import pyspark.sql.functions as F
import pyspark.ml.feature as mlf

from.config import SAMPLES_PER_FEATURE


def reduce_dimensionality(df: pys.DataFrame,
                          pred_cols: List[str],
                          feature_col: str,
                          label_col: str,
                          col_num: Optional[int] = None) \
        -> Tuple[pys.DataFrame, float]:
    r"""Use PCA to replace features with smaller vector.

    Sometimes need to reduce number of inputted predictors when we end up
    subsetting or subsampling. Using PCA right now but it needs to be moved
    to some sort of discriminant analysis. PCA is not the right long term
    approach

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe in question. Must have all pred_cols and assembled
        features called feature_col. Other cols will pass through
    pred_cols : list[str]
        array of colnames assembled in feature_col
    feature_col : str
        the name of the assembled feature col
    col_num : int, optional
        number of components desired. default value is count/


    Returns
    -------
    df_out
        df with modified features col
    explained_var : float
        variances explained by chosen number of components

    Raises
    ------
    AssertionError
        if feature col is not in df
        if pca name conflicts ( to be fixed )
        if df is too small to have a feature

    Uncaught Failures:
        if df is small but col_num is specified
        if SAMPLES_PER_FEATURE is 0 or invalid type or negative
        invalid pca values
        dumb values for col_num (not whole)
    Examples
    --------
    df, explained_var = reduce_dimensionality(df, pred_cols, feature_col, col_num)

    """
    if col_num is None:
        col_num = floor(df.where(F.col(label_col) == 1).count() / SAMPLES_PER_FEATURE)
        assert col_num > 0, "df is too small for a feature"

        # shouldnt occur outside test situations
        if col_num > len(pred_cols):
            col_num = len(pred_cols)

    assert feature_col in df.columns, "feature_col {fc} not in df columns".format(fc=feature_col)
    # FUTURE pick alternate colname instead of failing out
    pca_col = 'pca_' + feature_col
    assert pca_col not in df.columns, "pca_col name conflict".format(fc=feature_col)

    pca_estimator = mlf.PCA(k=col_num, inputCol=feature_col, outputCol='pca_' + feature_col)
    pca_model = pca_estimator.fit(df)
    df.cache()
    out_df = pca_model.transform(df)
    out_df = out_df.drop(feature_col).withColumnRenamed(pca_col, feature_col)
    out_df.cache()

    explained_var = float(sum(pca_model.explainedVariance))

    return out_df, explained_var
