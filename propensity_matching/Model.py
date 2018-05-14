"""Defines PropensityModel"""
import pyspark.ml as ml

from .evaluate import evaluate as _evaluate
from .impact import impact as _impact
from .transform import transform as _transform


class PropensityModel(ml.Model):
    r"""The entry point for transform, impact, and evaluate workflows.

    Parameters / Attributes
    ----------
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
    impact(df, matched_treatment, matched_control)
        Change the photo's gamma exposure.
    evaluate(pre_df, post_df, transform_df, by_col_group)
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

    def transform(self, df):
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
    matched_treatment : pyspark.sql.DataFrame
        matched rows in class 1
    matched_control : pyspark.sql.DataFrame
        matched rows in class 0

    Raises
    ------

    See Also
    --------
    transform in transform.py
    """
        matched_treatment, matched_control = _transform(df, self.prob_mod)
        return matched_treatment, matched_control

    def determine_impact(self, matched_treatment, matched_control):
        r"""Calculates effect of label col on response col, controlling
        for covariates

        Parameters
        ----------
        matched_treatment : pyspark.sql.DataFrame
            matched rows in class 1
        matched_control : pyspark.sql.DataFrame
            matched rows in class 0


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
        in_df = matched_treatment.union(matched_control.select(matched_treatment.columns))
        in_df.cache()
        label_col = self.prob_mod.getOrDefault('labelCol')
        response_col = self.response_col
        pred_cols_coefficients = zip(self.prob_mod.pred_cols, self.prob_mod.coefficients)

        treatment_rate, control_rate, adjusted_response = _impact(in_df, label_col, response_col, pred_cols_coefficients)
        return treatment_rate, control_rate, adjusted_response

    def evaluate_performance(self, pre_df, post_df, transform_df, by_col_group=True):
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
        by_col_group : bool = False, optional
            groups feature coefficients by source, as determined by prefix
            split by `GROUPED_COL_SEPARATOR` in config.py, into groups.
            An additional dict will be returned with the sum of the absolute
            value of the coeffs for each feature in a group. Because
            individual columns may vary from day to day, this is intended
            to verify model stability. not informative if user does not
            normalize variables beforehand 

        Returns
        -------
        performance_summary
            instance of PerformanceSummary class


        Raises
        ------
        
        See Also
        --------
        evaluate in evaluate.py


        Examples
        --------

        """
        performance_summary = _evaluate(self.prob_mod, pre_df, post_df, self.test_set, transform_df, by_col_group)
        return performance_summary
