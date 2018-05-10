import math

from . import Model

import pyspark.sql.functions as F
import pyspark.sql.types as T

import pyspark.ml as ml
import pyspark.ml.classification as mlc
import pyspark.ml.evaluation as mle
import pyspark.ml.feature as mlf
import pyspark.ml.regression as mlr
import pyspark.ml.tuning as mlt

import warnings

import pandas as pd

from .config import SAMPLES_PER_FEATURE

class Estimator(ml.Estimator):
    default_propensity_estimator_args = {
        "featuresCol": "features",
        "labelCol": "label",
        "predictionCol": "prediction",
        "maxIter": 10,
        "regParam": .2,
        "elasticNetParam": 0.5,
        # tol":1e-6,
        "fitIntercept": True,
        #"threshold":0.5,
        #"thresholds":None,
        "probabilityCol": "probability",
        #rawPredictionCol":"rawPrediction",
        #"standardization":True,
        #"weightCol":None,
        #aggregationDepth":2,
        "family": "binomial"
        }

    fit_data_prep_args = {
        'subsample_ratio': .1,
        'class_balance': 1,
        'train_prop': .8
        }

    def __init__(self,
                 pred_cols,
                 fit_data_prep_args=fit_data_prep_args,
                 propensity_estimator_args=default_propensity_estimator_args,
                 propensity_estimator=mlc.LogisticRegression,
                 response_col='response',
                 ):
        self.pred_cols = pred_cols
        self.fit_data_prep_args = fit_data_prep_args
        self.propensity_estimator_args = propensity_estimator_args
        self.propensity_estimator = propensity_estimator
        self.response_col = response_col

    def fit(self, df):

        # data prep
        self.df = df
        self.rebalance_df()
        self.split_test_train()

        # predictive model prep
        self.prepare_propensity_model()
        self.assemble_model_args()
        model = Model.Model(**self.model_args)
        return model

    # possible that desired class balance is not possible. in this case, take maximum possible
    # throw warning, reassign class balance
    def rebalance_df(self):
        num_1 = self.df.where(F.col(self.propensity_estimator_args['labelCol']) == 1).count()
        num_0 = self.df.where(F.col(self.propensity_estimator_args['labelCol']) == 0).count()

        max_ratio = num_0/num_1
        if self.fit_data_prep_args['class_balance'] > max_ratio:
            warnings.warn("Maximum class balance is {max_ratio} but requested is {class_balance} \
            Changing to {max_ratio}".format(max_ratio=max_ratio, class_balance=self.fit_data_prep_args['class_balance']))
            self.fit_data_prep_args['class_balance'] = max_ratio
            self.rebalanced_df = self.df
            return True
        desired_num_0 = self.fit_data_prep_args['class_balance']*num_1
        sample_frac_0 = desired_num_0/num_0
        rebalanced_df_0 = self.df.where(F.col(self.propensity_estimator_args['labelCol']) == 0).sample(False, sample_frac_0, 42)
        rebalanced_df = rebalanced_df_0.select(self.df.columns).union(self.df.where(F.col(self.propensity_estimator_args['labelCol']) == 1))
        self.rebalanced_df = rebalanced_df
        return True

    def subsample(self):
        self.subsampled_df = self.rebalanced_df.sample(False, self.fit_data_prep_args['subsample_ratio'], 42)
        return True

    def split_test_train(self):
        self.train_set, self.test_set = self.rebalanced_df.randomSplit([self.fit_data_prep_args['train_prop'], 1-self.fit_data_prep_args['train_prop']])
        return True

    def prepare_propensity_model(self):
        propensity_model = self.propensity_estimator(**self.propensity_estimator_args).fit(self.train_set)

        #pick fewer predictors in case of overfit
        train_count = self.train_set.count()
        if train_count < len(self.pred_cols) * SAMPLES_PER_FEATURE:
            num_cols = math.floor(train_count / SAMPLES_PER_FEATURE)
            if num_cols < 5:
                num_cols = 5
            cols = list(zip(self.pred_cols, propensity_model.coefficients))
            cols.sort(key=lambda x: -abs(x[1]))
            pred_cols = [x[0] for x in cols[0:num_cols]]
            features_col = self.propensity_estimator_args['featuresCol']
            refeaturizer = mlf.VectorAssembler(inputCols=pred_cols, outputCol=features_col)
            self.df = refeaturizer.transform(self.df.drop(features_col))
            self.train_set = refeaturizer.transform(self.train_set.drop(features_col))
            self.test_set = refeaturizer.transform(self.test_set.drop(features_col))
            propensity_model = self.propensity_estimator(**self.propensity_estimator_args).fit(self.train_set)
            self.pred_cols = pred_cols

        self.propensity_model = propensity_model
        self.propensity_model.pred_cols = self.pred_cols
        return True

    def assemble_model_args(self):
        self.model_args = {
            'prop_mod': self.propensity_model,
            'df': self.df,
            'train_set': self.train_set,
            'test_set': self.test_set,
            'response_col': self.response_col,
            'pred_cols': self.pred_cols
            }
