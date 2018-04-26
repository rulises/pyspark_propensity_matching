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
        # rawPredictionCol":"rawPrediction",
        #"standardization":True,
        #"weightCol":None,
        # aggregationDepth":2,
        "family": "binomial"
    }

    default_args = {
        'subsample_ratio': .1,
        'class_balance': 1,
        'train_prop': .8,
        'propensity_estimator_args': default_propensity_estimator_args,
        'propensity_estimator': mlc.LogisticRegression,
        'label_col': 'label',
        'feature_col': 'features',
        'probabilty_col': 'probability',
        'grain': .1,
        'response_col': 'response',
        'propensity_model': None,
        'pred_cols': None
    }

    def __init__(self, **kwargs):

        # set defaults
        for arg, value in self.default_args.items():
            setattr(self, arg, value)

        # set given args (overriding defaults)
        for arg, value in kwargs.items():
            if arg in self.default_args:
                setattr(self, arg, value)
            else:
                warnings.warn('unrecognized arg: {0}'.format(arg))

        self.propensity_estimator_args['probabilityCol'] = self.probabilty_col
        self.propensity_estimator_args['labelCol'] = self.label_col
        self.propensity_estimator_args['featuresCol'] = self.feature_col
        # set other args with defaults
        remaining_args = set(self.default_args.keys()) - set(kwargs.keys())
        for arg in remaining_args:
            setattr(self, arg, self.default_args[arg])
        assert self.pred_cols is not None, "user must provide prediction column names"

    def fit(self, df):

        # data prep

        self.df = df
        self.rebalance_df()
        self.subsample()
        self.split_test_train()

        # predictive model prep
        if self.propensity_model is None:
            self.prepare_propensity_model()
        self.assemble_model_args()
        model = Model.Model(**self.model_args)

        return model

    # possible that desired class balance is not possible. in this case, take maximum possible
    # throw warning, reassign class balance
    def rebalance_df(self):
        num_1 = self.df.where(F.col(self.label_col) == 1).count()
        num_0 = self.df.where(F.col(self.label_col) == 0).count()

        max_ratio = num_0/num_1
        if self.class_balance > max_ratio:
            warnings.warn("Maximum class balance is {max_ratio} but requested is {class_balance} \
            Changing to {max_ratio}".format(max_ratio=max_ratio, class_balance=self.class_balance))
            self.class_balance = max_ratio
            self.rebalanced_df = self.df
            return True
        desired_num_0 = self.class_balance*num_1
        sample_frac_0 = desired_num_0/num_0
        rebalanced_df_0 = self.df.where(F.col(self.label_col) == 0).sample(False, sample_frac_0, 42)
        rebalanced_df = rebalanced_df_0.select(self.df.columns).union(self.df.where(F.col(self.label_col) == 1))
        self.rebalanced_df = rebalanced_df
        return True

    def subsample(self):
        self.subsampled_df = self.rebalanced_df.sample(False, self.subsample_ratio, 42)
        return True

    def split_test_train(self):
        self.train_set, self.test_set = self.subsampled_df.randomSplit([self.train_prop, 1-self.train_prop])
        return True

    def prepare_propensity_model(self):
        self.propensity_model = self.propensity_estimator(**self.propensity_estimator_args).fit(self.train_set)
        return True

    def assemble_model_args(self):
        self.model_args = {
            'propmod': self.propensity_model,
            'subsample_ratio': .1,
            'class_balance': 1,
            'train_prop': .8,
            'propensity_estimator_args': self.default_propensity_estimator_args,
            'propensity_estimator': mlc.LogisticRegression,
            'label_col': self.label_col,
            'feature_col': self.feature_col,
            'probability_col': self.probabilty_col,
            'df': self.df,
            'train_set': self.train_set,
            'test_set': self.test_set,
            'grain': self.grain,
            'response_col': self.response_col,
            'pred_cols': self.pred_cols
        }
