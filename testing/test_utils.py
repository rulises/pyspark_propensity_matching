"""Testing module dedicated to utils.py from propensity_matching."""
# flake8: noqa E302, E402
import sys
import os

import pyspark.sql.functions as F

import pytest

sys.path.append(os.path.abspath('../'))

from propensity_matching.utils import reduce_dimensionality
from propensity_matching.config import MINIMUM_DF_COUNT, MINIMUM_POS_COUNT, SAMPLES_PER_FEATURE 
pytestmark = pytest.mark.usefixtures("spark", "df", "pred_cols", "feature_col", "label_col")


# reduce_dimensionality
#   wad
def test_rd_wad(df, pred_cols, feature_col, label_col):
    df, explained_variance = reduce_dimensionality(df, pred_cols, feature_col, label_col)
    return True

#   feature_col no in df columns
def test_rd_feat_col_missing(df, pred_cols, feature_col, label_col):
    feature_col = 'doesnt_exist'
    with pytest.raises(AssertionError, message="feature_col {fc} not in df columns".format(fc=feature_col)):
        df, explained_variance = reduce_dimensionality(df, pred_cols, feature_col, label_col)
    return True

#   pca_col name conflict
def test_rd_pca_colname_conflict(df, pred_cols, feature_col, label_col):
    df = df.withColumn('pca_' + feature_col, F.lit(1))
    with pytest.raises(AssertionError, message="pca_col name conflict"):
        df, explained_variance = reduce_dimensionality(df, pred_cols, feature_col, label_col)
    return True

#   df too small, num_cols not specified
def test_rd_df_too_small(df, pred_cols, feature_col, label_col):
    count = df.count()
    frac = SAMPLES_PER_FEATURE * .75 / count
    df = df.sample(fraction=frac, seed=42)
    with pytest.raises(AssertionError, message="df is too small for a feature"):
        df, explained_variance = reduce_dimensionality(df, pred_cols, feature_col, label_col)

#  min size allowed
def test_rd_min_df_size(df, pred_cols, feature_col, label_col):
    sample_bands = {}
    sample_1_frac = MINIMUM_POS_COUNT / df.where(F.col(label_col) == 1).count() + .01
    sample_bands[1] = sample_1_frac
    sample_0_frac = (MINIMUM_DF_COUNT - MINIMUM_POS_COUNT) / df.where(F.col(label_col) == 0).count() + .01
    sample_bands[0] = sample_0_frac

    df = df.sampleBy(col=label_col, fractions=sample_bands, seed=42)
    df, explained_variance = reduce_dimensionality(df, pred_cols, feature_col, label_col)
    return True
