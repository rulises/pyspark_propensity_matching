from collections import defaultdict
from typing import Type, Optional, Tuple

import pandas as pd
import numpy as np

import pyspark
from pyspark.ml import classification as mlc
import pyspark.sql.functions as F
dataframe = pyspark.sql.DataFrame

from .config import GROUPED_COL_SEPARATOR

# UPGRADE TO DATA CLASS WHEN DATABRICKS SUPPORT 3.7+
# maybe makes these named tuples
# currently classes to match spark ModelSummary behavior


class PropensityModelPerformanceSummary():
    def __init__(
                self,
                auc: float,
                informativeness: float,
                threshold: float,
                precision: float,
                recall: float,
                ModelSummary: Type[mlc.BinaryLogisticRegressionSummary]
                ):
        self.auc = auc
        self.informativeness = informativeness
        self.threshold = threshold
        self.precision = precision
        self.recall = recall
        self.ModelSummary = ModelSummary


class PerformanceSummary():
    def __init__(
            self,
            train_prop_mod_perf: Type[PropensityModelPerformanceSummary],
            test_prop_mod_perf, #: Optional[Type[PropensityModelPerformanceSummary]], weird error
            transform_prop_mod_perf, #: Optional[Type[PropensityModelPerformanceSummary]],
            model_weights: dict,
            model_weights_grouped: Optional[dict],
            bias_df: pd.DataFrame,
            total_bias_reduced: float):
        self.train_prop_mod_perf = train_prop_mod_perf
        self.test_prop_mod_perf = test_prop_mod_perf
        self.transform_prop_mod_perf = transform_prop_mod_perf
        self.model_weights = model_weights
        self.model_weights_grouped = model_weights_grouped
        self.bias_df = bias_df
        self.total_bias_reduced = total_bias_reduced


def evaluate(prop_mod: pyspark.ml.classification.LogisticRegressionModel,
             pre_df: dataframe,
             post_df: dataframe,
             test_df: Optional[dataframe] = None,
             transform_df: Optional[dataframe] = None,
             by_col_group: bool = False,) -> PerformanceSummary:

    train_prop_mod_perf, test_prop_mod_perf, transform_prop_mod_perf, prop_mod_weights, prop_mod_group_weights = \
        _eval_propensity_model(prop_mod, test_df, transform_df, by_col_group)

    label_col = prop_mod.getOrDefault('labelCol')
    cols = prop_mod.pred_cols + [label_col]
    bias_df, total_bias_reduced = _eval_match_performance(pre_df.select(cols), post_df.select(cols), label_col)

    performance_summary = PerformanceSummary(
        train_prop_mod_perf,
        test_prop_mod_perf,
        transform_prop_mod_perf,
        prop_mod_weights,
        prop_mod_group_weights,
        bias_df,
        total_bias_reduced
    )
    return performance_summary


def _eval_propensity_model(prop_mod: Type[pyspark.ml.Model],
                           test_df: Optional[dataframe],
                           transform_df: Optional[dataframe],
                           by_col_group: bool):

    train_prop_mod_perf = _eval_df_model(prop_mod.summary)

    if test_df is None:
        test_prop_mod_perf = None
    else:
        test_summary = prop_mod.evaluate(test_df)
        test_prop_mod_perf = _eval_df_model(test_summary)

    if transform_df is None:
        transform_prop_mod_perf = None
    else:
        non_pred_cols = [prop_mod.getOrDefault('predictionCol'),
                         prop_mod.getOrDefault('probabilityCol'),
                         prop_mod.getOrDefault('rawPredictionCol')
                         ]
        transform_df = transform_df.select([x for x in transform_df.columns if x not in non_pred_cols])
        transform_summary = prop_mod.evaluate(transform_df)
        transform_prop_mod_perf = _eval_df_model(transform_summary)

    prop_mod_weights = _eval_propensity_model_weights(prop_mod, by_col_group=False)
    if by_col_group:
        prop_mod_group_weights = _eval_propensity_model_weights(prop_mod, by_col_group=True)
    else:
        prop_mod_group_weights = None

    # awkward output consider named tuple?
    return train_prop_mod_perf, test_prop_mod_perf, transform_prop_mod_perf,\
        prop_mod_weights, prop_mod_group_weights


def _eval_df_model(summary: Type[mlc.LogisticRegressionSummary]) -> PropensityModelPerformanceSummary:
    auc = summary.areaUnderROC
    informativeness = summary.fMeasureByThreshold.groupby().max('F-Measure').collect()[0][0]
    threshold = summary.fMeasureByThreshold.where(F.col('F-Measure') == informativeness).collect()[0].threshold
    precision = summary.precisionByThreshold.where(F.col('threshold') == threshold).collect()[0].precision
    recall = summary.recallByThreshold.where(F.col('threshold') == threshold).collect()[0].recall

    df_model_perf = PropensityModelPerformanceSummary(
        auc=auc,
        informativeness=informativeness,
        threshold=threshold,
        precision=precision,
        recall=recall,
        ModelSummary=summary)
    return df_model_perf


def _eval_propensity_model_weights(prop_mod: pyspark.ml.classification.LogisticRegressionModel,
                                   by_col_group=False) -> dict:
    coeffs = prop_mod.coefficients

    # pred_cols were assigned to model in INSERT HERE
    # not normally part of model object HACK
    pred_cols = prop_mod.pred_cols

    if by_col_group:
        pred_cols = [col.split(GROUPED_COL_SEPARATOR)[0] for col in pred_cols]
    zipped = zip(pred_cols, coeffs)
    prop_mod_group_weights = defaultdict(lambda: 0)
    for tup in zipped:
        prop_mod_group_weights[tup[0]] += abs(tup[1])
    return dict(prop_mod_group_weights)


def _eval_match_performance(pre_df: dataframe, post_df: dataframe, label_col) -> Tuple[pd.DataFrame, float]:
    stan_bias_red_df, total_bias_reduced = _calc_standard_bias_reduced(pre_df, post_df, label_col)
    bias_red_df = _calc_bias_reduced(pre_df, post_df, label_col)
    bias_df = bias_red_df.join(stan_bias_red_df, how='outer')
    return bias_df, total_bias_reduced


def _calc_bias(df: dataframe, label_col: str) -> pd.DataFrame:
    bias_df = df.groupby(label_col).mean().toPandas().transpose()
    bias_df['bias'] = bias_df[1] - bias_df[0]
    bias_df = bias_df.reset_index()
    bias_df['index'] = bias_df['index'].str.replace(r'avg\(', '').str.replace(r')', '')
    bias_df = bias_df.set_index('index')[['bias']]
    bias_df = bias_df.loc[bias_df.index !='label', :]
    return bias_df


def _calc_standard_bias(df: dataframe, label_col: str) -> pd.DataFrame:
    var_df = _calc_var(df, label_col)
    bias_df = _calc_bias(df, label_col)

    bias_red_df = var_df.join(bias_df)
    bias_red_df['denominator'] = np.sqrt((bias_red_df['var_1'] + bias_red_df['var_0'])/2)
    bias_red_df['standard_bias'] = bias_red_df['bias']/bias_red_df['denominator'] * 100
    bias_red_df = bias_red_df[['standard_bias']]
    return bias_red_df


def _calc_var(df: pyspark.sql.DataFrame, label_col: str) -> pd.DataFrame:
    pred_cols = [x for x in df.columns if x != label_col]
    s_var_df = df.groupby(label_col).agg({x: 'variance' for x in pred_cols}).toPandas().transpose()
    s_var_df = s_var_df.reset_index()
    s_var_df['index'] = s_var_df['index'].str.replace(r')', '').str.replace(r'variance\(', '')
    s_var_df = s_var_df.set_index('index')
    s_var_df.columns = ["var_{0}".format(x) for x in s_var_df.columns]
    s_var_df = s_var_df.loc[s_var_df.index !='label', :]
    return s_var_df


def _calc_bias_reduced(pre_df: dataframe, post_df: dataframe, label_col: str) -> pd.DataFrame:
    pre_bias = _calc_bias(pre_df, label_col)
    pre_bias.columns = ['pre_bias']

    post_bias = _calc_bias(post_df, label_col)
    post_bias.columns = ['post_bias']

    b_red_df = pre_bias.join(post_bias, how='outer')
    b_red_df['bias_reduced_absolute'] = b_red_df['pre_bias'] - b_red_df['post_bias']
    b_red_df['bias_reduced_relative'] = b_red_df['bias_reduced_absolute']/b_red_df['pre_bias'] * 100
    return b_red_df


def _calc_standard_bias_reduced(pre_df: dataframe, post_df: dataframe, label_col: str) -> Tuple[pd.DataFrame, float]:

    pre_standard_bias = _calc_standard_bias(pre_df, label_col)
    pre_standard_bias.columns = ['pre_standard_bias']

    post_standard_bias = _calc_standard_bias(post_df, label_col)
    post_standard_bias.columns = ['post_standard_bias']

    stan_bias_red_df = pre_standard_bias.join(post_standard_bias, how='outer')
    stan_bias_red_df['standard_bias_reduced_absolute'] = stan_bias_red_df['pre_standard_bias']\
        - stan_bias_red_df['post_standard_bias']
    stan_bias_red_df['standard_bias_reduced_relative'] = stan_bias_red_df['standard_bias_reduced_absolute']\
        / stan_bias_red_df['pre_standard_bias']*100

    total_bias_reduced = (1 - stan_bias_red_df.post_standard_bias.abs().sum()/stan_bias_red_df.pre_standard_bias.abs().sum())*100
    return stan_bias_red_df, float(total_bias_reduced)
