"""Defines PropensityModel"""
from typing import Tuple
from collections import namedtuple

from pyspark.sql import DataFrame
import pyspark.ml as ml


from .evaluate import evaluate as _evaluate
from .impact import impact as _impact
from .transform import transform as _transform


class PropensityModel():
    r"""The entry point for transform, impact, and evaluate workflows.

    Parameters / Attributes
    -----------------------
    prob_mod : pyspark.ml.classification.LogisticRegressionModel
        Model obj to predict probability of being in label class 1
        prob_mod.pred_cols houses feature columns names
        getters are also used to for label and assembled features col
    df : pyspark.sql.DataFrame
        The actual data
    train_set : pyspark.sql.DataFrame
        data used to train prob_mod
    test_set : pyspark.sql.DataFrame
        data used to test prob_mod
    response_col : str
        col holding the response variable

    Methods
    -------
    transform(df)
        Represent the photo in the given colorspace.
    determine_impact(df, matched_treatment, matched_control)
        Change the photo's gamma exposure.
    evaluate_performance(pre_df, post_df, transform_df, by_col_group)
        Change the photo's gamma exposure.
    """
    def __init__(self,
                 prob_mod,
                 df,
                 train_set,
                 test_set,
                 response_col,
                 ):

        self.prob_mod = prob_mod
        self.df = df
        self.train_set = train_set
        self.test_set = test_set
        self.response_col = response_col

    def transform(self,
                  df: DataFrame) ->Tuple[DataFrame, dict]:
        r"""A one-line summary that does not use variable names or the
        function name.

        Several sentences providing an extended description. Refer to
        variables using back-ticks, e.g. `var`.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            full dataframe to propensity_match on 
        Returns
        -------
        df: pyspark.sql.DataFrame
            matched observations
        match_info : dict
            depending on matching, contains information about the match

        Raises
        ------
        UncaughtExceptiosn

        See Also
        --------
        transform in transform.py
        """
        df, match_info= _transform(df, self.prob_mod)
        return df, match_info

    def determine_impact(self,
                         df: DataFrame)-> Tuple[float, float, float]:
        r"""Calculates effect of label col on response col, controlling
        for covariates

        Parameters
        ----------
        df : pyspark.sql.DataFrame


        Returns
        -------
        treatment_rate : float
            % of matched class 1s that have response 1 (as opposed to 0)
        control_rate : float
            % of matched class 1s that have response 1 (as opposed to 0)
        adjusted_response : float
            impact of label on reponse col, with further adjustments for bias

        Raises
        ------

        See Also
        --------
        impact in impact.py

        Examples
        --------
        """

        treatment_rate, control_rate, adjusted_response = _impact(df=df,
                                                                  response_col=self.response_col,
                                                                  prob_mod=self.prob_mod)
        return treatment_rate, control_rate, adjusted_response

    def evaluate_performance(self,
                             pre_df,
                             post_df,
                             transform_df)-> namedtuple:
        r"""provides goodness metrics for propensity match

        Considers both the probability model as well as the matching itself


        Parameters
        ----------
        pre_df : pyspark.sql.DataFrame
            dataframe before the propensity matching. used to calculate 
            starting standard bias
        post_df : pyspark.sql.DataFrame
            dataframe after propensity matching. used to calculate ending
            standard bias
        transform_df : pyspark.sql.DataFrame
            df transformed by probability model. used to calculate model
            goodness metrics on whole dataframe, as opposed to class 
            balances test and train sets

        Returns
        -------
        performance_summary : namedtuple
            'test_prob_mod_perf': propensity_model_performance_summary
            'train_prob_mod_perf' : propensity_model_performance_summary
            'transform_prob_mod_perf' : propensity_model_performance_summary
            'bias_df': pd.DataFrame
                for each col has pre, post, absolute reduce, relative
                reduced bias
            'total_bias_reduced': float
                1 - (sum postbias of features/ sum rebias of features)
            'starting_bias_mean': float
                mean of prebias
            'starting_bias_var': float
                var of prebias
            where
                propensity_model_performance_summary : namedtuple
                    'auc' : float
                    'auprc' : float
                        area under precision recall curve
                    'threshold' : float
                    'informativeness' (f1) : float
                    'precision' : float
                    'recall' : float
                    'accuracy'  : float



        Raises
        ------
        
        See Also
        --------
        evaluate in evaluate.py


        Examples
        --------

        """
        performance_summary = _evaluate(prob_mod=self.prob_mod,
                                        pre_df=pre_df,
                                        post_df=post_df,
                                        test_df=self.test_set,
                                        train_df=self.train_set,
                                        transform_df=transform_df)
        return performance_summary
