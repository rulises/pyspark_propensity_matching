import math
from typing import Tuple

import numpy as np
import pandas as pd

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.ml.classification as mlc

from .config import GRAIN, QUANTILE_ERROR_TOLERANCE, UTIL_BOOST_THRESH_1, UTIL_BOOST_THRESH_2, UTIL_BOOST_THRESH_3

dataframe = pyspark.sql.DataFrame


def score_df(df: dataframe, prop_mod: mlc.LogisticRegressionModel) -> Tuple[dataframe, dataframe, str]:
    scored_df = prop_mod.transform(df)
    prob_col = prop_mod.getOrDefault('probabilityCol')
    prob_1_col = prob_col + "_1"
    scored_df = scored_df.withColumn(prob_1_col, F.udf(lambda x: float(x[0]), T.FloatType())(prob_col))
    label_col = prop_mod.getOrDefault('labelCol')

    treatment_df = scored_df.where(F.col(label_col) == 1)
    control_can_df = scored_df.where(F.col(label_col) == 0)
    return treatment_df, control_can_df, prob_1_col


def make_match_col(t_df: dataframe, c_can_df: dataframe, grain: float, metric_col: str) -> Tuple[dataframe, dataframe]:

    probs = list(np.arange(0, 1, grain))
    probs = [float(x) for x in probs]

    quantiles = t_df.approxQuantile(col=metric_col, probabilities=probs, relativeError=QUANTILE_ERROR_TOLERANCE)

    def make_udf(quantiles):
        quantiles = np.array(quantiles)
        group_dict = dict()
        for num in np.arange(0, 1.01, .01):
            round_num = int(round(num*100, 0))
            group_dict[round_num] = (num > quantiles).sum()

        def udf(number):
            return int(group_dict.get(int(round(number*100, 0)), -1))
        return udf

    match_col_udf = F.udf(make_udf(quantiles), T.IntegerType())

    round_col = metric_col + "_round"

    t_df = t_df.withColumn(round_col, F.round(t_df[metric_col], 2))
    c_can_df = c_can_df.withColumn(round_col, F.round(c_can_df[metric_col], 2))

    t_df = t_df.withColumn('match_col', match_col_udf(round_col))
    c_can_df = c_can_df.withColumn('match_col', match_col_udf(round_col))
    return t_df, c_can_df


def calc_sample_fracs(t_df: dataframe, c_can_df: dataframe) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    t_counts = t_df.groupby(['match_col']).count().withColumnRenamed('count', 'treatment')
    c_can_counts = c_can_df.groupby(['match_col']).count().withColumnRenamed('count', 'control')
    fracs = t_counts.join(c_can_counts, on=['match_col'])
    fracs = fracs.toPandas()
    sample_fracs, scale, drop = calc_optimal_subset(fracs)
    return sample_fracs[['match_col', 'treatment_scaled_sample_fraction']],\
           sample_fracs[['match_col', 'control_scaled_sample_fraction']],\
           scale, drop


def calc_optimal_subset(fracs: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    fracs = fracs.copy(deep=True)
    fracs['control_sample_fraction_naive'] = fracs['treatment']/fracs['control']
    scale_factor = fracs.control_sample_fraction_naive.max()**-1

    # if no subscaling is necessary return fracs as is
    if scale_factor >= 1:
        fracs['control_scaled_sample_fraction'] = fracs['control_sample_fraction_naive']
        fracs['treatment_scaled_sample_fraction'] = 1
        fracs = fracs[['match_col', 'treatment_scaled_sample_fraction', 'control_scaled_sample_fraction']]
        return fracs, 1, 0

    options = create_options_grid(fracs, scale_factor)
    options['utility'] = options.apply(calc_util_wrapper, axis=1)

    # pick best
    max_util = options.utility.max()
    best_row = options[options.utility == max_util]
    if len(best_row) > 1:
        best_row = best_row.iloc[0]
    winning_scale = best_row['scale'][0]
    winning_drop = best_row['percent_dropped'][0]

    fracs['control_scaled_sample_fraction'] = np.min([(fracs['treatment'] * winning_scale/fracs['control']).values, [1]*len(fracs)], axis=0)
    fracs['treatment_scaled_sample_fraction'] = fracs['control_scaled_sample_fraction'] * fracs['control']/fracs['treatment']
    fracs = fracs[['match_col', 'treatment_scaled_sample_fraction', 'control_scaled_sample_fraction']]

    # return fracs
    return fracs, winning_scale, winning_drop


def create_options_grid(fracs: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    fracs = fracs.copy(deep=True)
    scales = np.linspace(1, scale_factor, num=100, endpoint=True)
    options = pd.DataFrame(columns=['scale', 'percent_dropped', 'number'])

    for scale in scales:
        # calc new frac samples, maximum of 1
        fracs['control_scaled_sample_fraction'] = np.min([(fracs['treatment'] * scale/fracs['control']).values, [1]*len(fracs)], axis=0)
        fracs['treatment_scaled_sample_fraction'] = fracs['control_scaled_sample_fraction'] * fracs['control']/fracs['treatment']

        # calc %drop as difference of scale and actual ( e.g. where we pinned max at 1 in control scaled sample fraction)
        num_dropped = (fracs['treatment'] * (([scale] * len(fracs)) - fracs['treatment_scaled_sample_fraction'])).sum()
        percent_dropped = num_dropped/(fracs['treatment'] * scale).sum()

        # calc new total
        number = (fracs['treatment']*fracs['treatment_scaled_sample_fraction']).sum()
        options = options.append({'scale': scale, 'percent_dropped': percent_dropped, 'number': number}, ignore_index=True)

    return options


def calc_util_wrapper(row):
    return calc_util(row['number'], row['percent_dropped'])


def calc_util(number, dropped):
    log_value = math.log10(number/1000 + 1)
    threshold_boost = logistic_function(L=math.log10(number / UTIL_BOOST_THRESH_1 + 1) / 10, x=number, x0=UTIL_BOOST_THRESH_1) \
                    + logistic_function(L=math.log10(number / UTIL_BOOST_THRESH_2 + 1) / 10, x=number, x0=UTIL_BOOST_THRESH_2) \
                    + logistic_function(L=math.log10(number / UTIL_BOOST_THRESH_3 + 1) / 10, x=number, x0=UTIL_BOOST_THRESH_3)
    dropped_penalty = 1-min(math.exp(dropped)-1, 1)
    utility = dropped_penalty * (log_value + threshold_boost)
    return utility


def logistic_function(x, L, k=1, x0=0):
    try:
        return L / (1 + math.exp(-k * (x - x0)))
    except OverflowError:
        if x >= x0:
            return L
        if x < x0:
            return 0


def sample_dfs(t_df: pyspark.sql.DataFrame, t_fracs: pd.DataFrame, c_can_df: pyspark.sql.DataFrame, c_fracs: pd.DataFrame):
        t_fracs = t_fracs.set_index('match_col').treatment_scaled_sample_fraction.to_dict()
        t_dict = {}
        for key, value in t_fracs.items():
            t_dict[int(key)] = min(float(value), 1)
        t_out = t_df.sampleBy(col='match_col', fractions=t_dict, seed=42)

        c_fracs = c_fracs.set_index('match_col').control_scaled_sample_fraction.to_dict()
        c_dict = {}
        for key, value in c_fracs.items():
            c_dict[int(key)] = float(value)
        c_out = c_can_df.sampleBy(col='match_col', fractions=c_dict, seed=42)
        return t_out, c_out


def match(df_in, prop_mod):
    t_df, c_can_df, metric_col = get_metric(df_in, prop_mod)
    t_df, c_can_df = make_match_col(t_df, c_can_df, GRAIN, metric_col)
    t_fracs, c_fracs, scaled, dropped = calc_sample_fracs(t_df, c_can_df)
    t_out, c_out = sample_dfs(t_df, t_fracs, c_can_df, c_fracs)
    t_out.cache()
    c_out.cache()
    return t_out, c_out


def get_metric(df_in, prop_mod):
    t_df, c_can_df, metric_col = score_df(df_in, prop_mod)
    return t_df, c_can_df, metric_col


def transform(df_in, prop_mod):
    matched_treatment, matched_control = match(df_in, prop_mod)
    return matched_treatment, matched_control
