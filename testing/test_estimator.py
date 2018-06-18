"""Testing module dedicated to estimator.py from propensity_matching."""
# pydocstyle: noqa
# flake8: noqa E302, E402
import sys
import os

from math import floor, ceil
import pyspark.sql.functions as F
import pyspark.ml.classification as mlc

import pytest

sys.path.append(os.path.abspath('../'))

from propensity_matching.config import MINIMUM_DF_COUNT, MINIMUM_POS_COUNT, SAMPLES_PER_FEATURE
from propensity_matching.estimator import PropensityEstimator
pytestmark = pytest.mark.usefixtures("spark", "df", "pred_cols", "feature_col", "label_col", 'response_col')

# __init__
#   wad
def test_init_wad(pred_cols, response_col):
    pe = PropensityEstimator(pred_cols=pred_cols, response_col=response_col)
    return True

#   prob estimator not lr
def test_init_bad_lr(pred_cols, response_col):
    with pytest.raises(NotImplementedError, message='only pyspark.ml.classification.LogisticRegression is currently supported'):
        pe = PropensityEstimator(pred_cols=pred_cols, response_col=response_col, probability_estimator=mlc.NaiveBayes)
    return True

@pytest.fixture(scope='module')
def default_pe(pred_cols, response_col):
    pe = PropensityEstimator(pred_cols=pred_cols, response_col=response_col)
    return pe

# fit
#   min df count
def test_fit_min_df_count(spark, df, default_pe):
    df = spark.createDataFrame(df.take(MINIMUM_DF_COUNT - 1))
    with pytest.raises(AssertionError, message='df is too small to fit model'):
        default_pe.fit(df)
    return True


#   min pos count
def test_fit_min_pos_count(spark, df, default_pe, label_col):
    df_neg = df.where(F.col(label_col) == 0)
    df_pos = spark.createDataFrame(df.where(F.col(label_col) == 1).take(MINIMUM_POS_COUNT - 1))
    df = df_neg.union(df_pos.select(df_neg.columns))
    with pytest.raises(AssertionError, message='not enough positive samples in df to fit'):
        model = default_pe.fit(df)
    return True

#   reduce dimensionality
def test_fit_rd(spark, df , default_pe, pred_cols, label_col):
    count = df.count()
    pos_count = df.where(F.col(label_col)==1).count()
    neg_count = count - pos_count
    pos_ratio = pos_count / count
    
    num_needed = (len(pred_cols) - 1) * SAMPLES_PER_FEATURE

    pos_needed = max(MINIMUM_POS_COUNT+1, ceil(pos_ratio * num_needed))
    neg_needed = num_needed - pos_needed

    pos_df = spark.createDataFrame(df.where(F.col(label_col) == 1).take(pos_needed))
    neg_df = spark.createDataFrame(df.where(F.col(label_col) == 0).take(neg_needed))

    df = pos_df.union(neg_df.select(pos_df.columns))
    default_pe.fit(df)
    return True
#   wad
def test_fit(spark, df, default_pe):
    default_pe.fit(df)
    return True

# _rebalance_df
#   more 1 than 0
def test_rebalance_more1(df, label_col, default_pe):
    df = df.replace(to_replace={1:0, 0:1}, value=-1, subset=[label_col])
    default_pe = default_pe
    default_pe.df = df
    with pytest.raises(NotImplementedError, message="class rebalanced not implemented for class 1 > class 0"):
        default_pe._rebalance_df()

#   ratio adjustment
def test_rebalance_ratioadj(df, label_col, default_pe, spark):
    n = 5
    default_pe = default_pe
    default_pe.fit_data_prep_args['class_balance'] = n

    pos_count = df.where(F.col(label_col) == 1).count()
    neg_needed = floor(pos_count * n / 2)
    df_neg = spark.createDataFrame(df.where(F.col(label_col) == 0).take(neg_needed))
    df_pos = df.where(F.col(label_col) == 1)
    df = df_neg.union(df_pos.select(df_neg.columns))
    default_pe.df = df    
    default_pe._rebalance_df()
    return True
#   wad
def test_rebalance_wad(df, default_pe):
    default_pe = default_pe
    default_pe.df = df
    default_pe._rebalance_df()
    return True


# _split_test_train
#   no tests. Modes of failure include: bad arg for train_prop. (not between 0 and 1)

# _prepare_probability_model
#   no tests. fails if there are invalid args for fitting or data errors