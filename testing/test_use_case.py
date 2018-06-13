import sys
import os
import datetime as dt

import pyspark

import pytest

sys.path.append(os.path.abspath('../'))

from propensity_matching.estimator import PropensityEstimator
pytestmark = pytest.mark.usefixtures("df")


def test_case(df):
    df = df
    non_pred_cols = ['label', 'response', 'features']
    pred_cols = [x for x in df.columns if x not in non_pred_cols]
    estimator = PropensityEstimator(pred_cols)
    model = estimator.fit(df)
    df = model.df
    t,c = model.transform(df)
    treatment_rate, control_rate, adjusted_response = model.determine_impact(t,c)
    post_df = t.union(c.select(t.columns))
    eval_out = model.evaluate_performance(df, post_df, post_df, True)
    return True
