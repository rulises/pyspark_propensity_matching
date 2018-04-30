
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

import numpy as np
import pandas as pd

import pyspark.ml as ml
import pyspark.ml.classification as mlc
import pyspark.ml.evaluation as mle
import pyspark.ml.feature as mlf
import pyspark.ml.regression as mlr
import pyspark.ml.tuning as mlt

import math
import warnings

from .Evaluate import evaluate as evaluate_
from .Impact import impact as impact_
from .Transform import transform as transform_


class Model(ml.Model):
    def __init__(self,
                 prop_mod,
                 df,
                 train_set,
                 test_set,
                 response_col,
                 pred_cols
                 ):

        self.prop_mod = prop_mod
        self.df = df
        self.train_set = train_set
        self.test_set = test_set
        self.response_col = response_col

    def transform(self, df):
        matched_treatment, matched_control = transform_(df, self.prop_mod)
        return matched_treatment, matched_control

    def impact(self, matched_treatment, matched_control):
        in_df = matched_treatment.union(matched_control.select(matched_treatment.columns))
        in_df.cache()
        label_col = self.prop_mod.getOrDefault('labelCol')
        response_col = self.response_col
        pred_cols_coefficients = zip(self.prop_mod.pred_cols, self.prop_mod.coefficients)

        treatment_rate, control_rate, adjusted_response = impact_(in_df, label_col, response_col, pred_cols_coefficients)
        return treatment_rate, control_rate, adjusted_response

    def evaluate(self, pre_df, post_df, transform_df, by_col_group=True):
        performance_summary = evaluate_(self.prop_mod, pre_df, post_df, self.test_set, transform_df, by_col_group)
        return performance_summary
