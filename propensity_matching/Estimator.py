"""module holding PropensityEstimator (ml.Estimator)."""
from typing import List, Optional, Type
from math import floor
import warnings

import pyspark.sql as pys
import pyspark.sql.functions as F
import pyspark.ml as ml
import pyspark.ml.classification as mlc

from .config import MINIMUM_DF_COUNT, MINIMUM_POS_COUNT, SAMPLES_PER_FEATURE
from .model import PropensityModel
from .utils import reduce_dimensionality


class PropensityEstimator(ml.Estimator):
    """
    ml.Estimator to fit and return a PropensityModel.

    Instance Attributes
    ----------
    pred_cols : List[str]
        list of columns used as predictors
    fit_data_prep_args : dict
        args for class balance and test/train split during regression fitting
    probability_estimator_args : dict
        args for regressio model
    probability_estimator : pyspark.ml.classification.LogisticRegression
        currently only supported probability estimator
    response_col : str = 'response'
        column containing response variable
    train_set : pyspark.sql.df
        training set used by probability estimator. created by _split_test_train
    test_set : pyspark.sql.df
        test set used by probability estimator. created by _split_test_train
    rebalanced_df : pyspark.sql.df
        dataframe with class balance given in fit_data_prep_args

    Class Attributes
    ----------
    default_probability_estimator_args
    default_fit_data_prep_args


    Methods
    -------
    __init__(pred_cols: List[str],
            fit_data_prep_args: dict = default_fit_data_prep_args,
            probability_estimator_args=default_probability_estimator_args,
            probability_estimator=mlc.LogisticRegression,
            response_col='response' )
        Represent the photo in the given colorspace.
    fit(df: pyspark.sql.DataFrame)
        return PropensityModel
    """

    default_probability_estimator_args = {
        "featuresCol": "features",
        "labelCol": "label",
        "predictionCol": "prediction",
        "maxIter": 10,
        "regParam": .2,
        "elasticNetParam": 0.5,
        # tol":1e-6,
        "fitIntercept": True,
        # "threshold":0.5,
        # "thresholds":None,
        "probabilityCol": "probability",
        # "rawPredictionCol":"rawPrediction",
        # "standardization":True,
        # "weightCol":None,
        # "aggregationDepth":2,
        "family": "binomial"
    }

    default_fit_data_prep_args = {
        'class_balance': 1,
        'train_prop': .8
    }

    def __init__(self,
                 pred_cols: List[str],
                 fit_data_prep_args: Optional[dict] = default_fit_data_prep_args,
                 probability_estimator_args: dict = default_probability_estimator_args,
                 probability_estimator: Type[ml.Estimator] = mlc.LogisticRegression,
                 response_col: str ='response',
                 ):
        self.pred_cols = pred_cols
        self.fit_data_prep_args = fit_data_prep_args
        self.probability_estimator_args = probability_estimator_args
        self.probability_estimator = probability_estimator
        if probability_estimator != mlc.LogisticRegression:
            raise NotImplementedError('only pyspark.ml.classification.LogisticRegression is currently supported')
        self.response_col = response_col

    def fit(self, df: pys.DataFrame) -> PropensityModel:
        """
        Fit propensity model and return.

        Must prepare df and fit probability model from estimator.
        df is rebalanced and, if necessary, features are adjusted.
        will fail if df is too small or has too few positive samples

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            dataframe containing desired data. Must have predictor columns as well as features, label column specificed in
            propensity_estimator_args and response col given in __init__

        Returns
        -------
        model: PropensityModel
            ml.Model object for propensity matching.

        Raises
        ------
        AssertionError
            df too small
            too few positive samples

        NotImplementedError
            model other than LogisticRegression

        Uncaught Errors:
            invalid param args.

        Examples
        --------
        model = propensity_estimator.fit(df)

        """
        df.cache()
        df_count = df.count()
        assert df_count > MINIMUM_DF_COUNT, "df is too small to fit model"
        pos_count = df.where(F.col(self.probability_estimator_args['labelCol']) == 1).count()
        assert pos_count > MINIMUM_POS_COUNT, "not enough positive samples in df to fit"

        col_num = floor(pos_count * self.fit_data_prep_args['train_prop'] / SAMPLES_PER_FEATURE)
        if col_num < len(self.pred_cols):
            df, explained_var = reduce_dimensionality(
                df=df,
                pred_cols=self.pred_cols,
                feature_col=self.probability_estimator_args['featuresCol'],
                label_col=self.probability_estimator_args['labelCol']
            )
        self.df = df

        self._rebalance_df()
        self._split_test_train()

        self._prepare_probability_model()
        model = PropensityModel(
            prob_mod=self.probability_model,
            df=self.df,
            train_set=self.train_set,
            test_set=self.test_set,
            response_col=self.response_col
        )
        return model

    def _rebalance_df(self) -> bool:
        """
        Create new df with forced class balance for label to help with training.

        Raises
        ------
        NotImplementedError
            where there is more of class 1 than class 0

        Uncaught Errors
            where class balance is less than 1

        """
        label_col = self.probability_estimator_args['labelCol']
        num_1 = self.df.where(F.col(label_col) == 1).count()
        num_0 = self.df.where(F.col(label_col) == 0).count()

        if num_1 > num_0:
            raise NotImplementedError("class rebalanced not implemented for class 1 > class 0")

        # should have already failed out in fit if num_1 would have returned 0
        max_ratio = num_0 / num_1
        # if desired class ratio is impossible, take max possible ratio, reassign class balance and warn
        if self.fit_data_prep_args['class_balance'] > max_ratio:
            warnings.warn("Maximum class balance is {max_ratio} but requested is {class_balance} \
            Changing to {max_ratio}".format(max_ratio=max_ratio, class_balance=self.fit_data_prep_args['class_balance']))
            self.fit_data_prep_args['class_balance'] = max_ratio

        desired_num_0 = self.fit_data_prep_args['class_balance'] * num_1
        sample_frac_0 = min(1, float(desired_num_0 / num_0))  # protect against non-float types (numpy) & floating point error
        rebalanced_df_0 = self.df.where(F.col(label_col) == 0).sample(withReplacement=False, fraction=float(sample_frac_0), seed=42)
        rebalanced_df = rebalanced_df_0.select(self.df.columns).union(self.df.where(F.col(label_col) == 1).select(self.df.columns))
        self.rebalanced_df = rebalanced_df
        return True

    def _split_test_train(self) -> bool:
        """Create test, train set attributes based on fit_data_prep_args."""
        self.train_set, self.test_set = self.rebalanced_df.randomSplit([self.fit_data_prep_args['train_prop'], 1 - self.fit_data_prep_args['train_prop']])
        return True

    def _prepare_probability_model(self):
        """Fit probability model, embed pred cols inside."""
        probability_model = self.probability_estimator(**self.probability_estimator_args).fit(self.train_set)
        # guard against overfit happened in fit before _rebalance_df and _split_test_train were called
        self.probability_model = probability_model
        self.probability_model.pred_cols = self.pred_cols
        return True
