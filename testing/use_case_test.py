
import sys
import os
import datetime as dt

import pyspark

import pytest

sys.path.append(os.path.abspath('../'))

import propensity_matching

pytestmark = pytest.mark.usefixtures("small_df", "med_df", "big_df")


def test_case_small(small_df):
    df = small_df
    non_pred_cols = ['label', 'response', 'features']
    pred_cols = [x for x in df.columns if x not in non_pred_cols]
    estimator = propensity_matching.Estimator.Estimator(pred_cols)
    model = estimator.fit(df)
    t,c = model.transform(df)
    treatment_rate, control_rate, adjusted_response = model.impact(t,c)
    post_df = t.union(c.select(t.columns))
    eval_out = model.evaluate(df, post_df, post_df, True)
    return True

def test_case_med(med_df):
    df = med_df
    non_pred_cols = ['label', 'response', 'features']
    pred_cols = [x for x in df.columns if x not in non_pred_cols]
    estimator = propensity_matching.Estimator.Estimator(pred_cols)
    model = estimator.fit(df)
    t,c = model.transform(df)
    treatment_rate, control_rate, adjusted_response = model.impact(t,c)
    post_df = t.union(c.select(t.columns))
    eval_out = model.evaluate(df, post_df, post_df, True)
    return True

def test_case_big(big_df):
    df = big_df
    non_pred_cols = ['label', 'response', 'features']
    pred_cols = [x for x in df.columns if x not in non_pred_cols]
    estimator = propensity_matching.Estimator.Estimator(pred_cols)
    model = estimator.fit(df)
    t,c = model.transform(df)
    treatment_rate, control_rate, adjusted_response = model.impact(t,c)
    post_df = t.union(c.select(t.columns))
    eval_out = model.evaluate(df, post_df, post_df, True)
    return True
