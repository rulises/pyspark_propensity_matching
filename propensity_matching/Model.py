
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


class Model(ml.Model):

    def __init__(self, **kwargs):
        for arg, value in kwargs.items():
            setattr(self, arg, value)
        return None

    # def transform(self, df = None):
    #     if df is None:
    #         df = self.df

    #     col = self.probability_col

    #     scored_df = self.propmod.transform(df)

    #     scored_df = scored_df.withColumn(col+"_1", F.udf(lambda x:float(x[0]), T.FloatType())(col))
    #     col = col+"_1"

    #     treatment_df = scored_df.where(F.col(self.label_col) == 1)

    #     self.treatment_count = treatment_df.count()

    #     control_can_df = scored_df.where(F.col(self.label_col) == 0)

    #     probs = list(np.arange(0, 1, self.grain))
    #     probs = [float(x) for x in probs]

    #     quantiles = treatment_df.approxQuantile(col=col, probabilities=probs, relativeError=.05)

    #     def make_udf(quantiles):
    #         quantiles = np.array(quantiles)
    #         group_dict = dict()
    #         for num in np.arange(0, 1.01, .01):
    #             num2 = int(round(num*100, 0))
    #             group_dict[num2] = (num>quantiles).sum()

    #         def udf(number):
    #             return int(group_dict.get(int(round(number*100,0)), -1))
    #         return udf

    #     match_col_udf = F.udf(make_udf(quantiles), T.IntegerType())

    #     treatment_df = treatment_df.withColumn(col+"_round", F.round(treatment_df[col],2))
    #     control_can_df = control_can_df.withColumn(col+"_round", F.round(control_can_df[col],2))

    #     treatment_df = treatment_df.withColumn('match_col', match_col_udf(col+"_round"))
    #     control_can_df = control_can_df.withColumn('match_col', match_col_udf(col+"_round"))

    #     treatment_counts = treatment_df.groupby(['match_col']).count().withColumnRenamed('count', 'treatment')
    #     control_counts = control_can_df.groupby(['match_col']).count().withColumnRenamed('count', 'control')
    #     fracs = treatment_counts.join(control_counts, on=['match_col'])
    #     fracs = fracs.withColumn('control_frac_1', fracs.treatment/fracs.control)
    #     #rebalance to guard against overflow
    #     fracs_pd = fracs.select(['match_col', 'control_frac_1']).toPandas()
    #     if fracs_pd['control_frac_1'].max()>.9:
    #       fracs_pd['control_frac_2'] = fracs_pd['control_frac_1']/(fracs_pd['control_frac_1'].max()/.9)
    #       fracs_pd['treatment_frac'] = fracs_pd['control_frac_2']/fracs_pd['control_frac_1']
    #     else:
    #       fracs_pd['control_frac_2'] = fracs_pd['control_frac_1']
    #       fracs_pd['treatment_frac'] = 1

    #     #sample both
    #     fracs_dict_control = fracs_pd[['match_col', 'control_frac_2']].set_index('match_col').control_frac_2.astype(dtype='float').to_dict()
    #     fracs_dict_control_float = dict()
    #     for key, value in fracs_dict_control.items():
    #         fracs_dict_control_float[float(key)] = float(value)
    #     control_df_out = control_can_df.sampleBy(col='match_col', fractions=fracs_dict_control_float, seed=42)

    #     fracs_dict_treatment = fracs_pd[['match_col', 'treatment_frac']].set_index('match_col').treatment_frac.astype(dtype='float').to_dict()
    #     fracs_dict_treatment_float = dict()
    #     for key, value in fracs_dict_treatment.items():
    #         fracs_dict_treatment_float[float(key)] = float(value)
    #     treatment_df_out = treatment_df.sampleBy(col='match_col', fractions=fracs_dict_treatment_float, seed=42)

    #     self.matched_treatment = treatment_df_out
    #     self.matched_control = control_df_out
    #     self.post_match_df = treatment_df_out.union(control_df_out.select(treatment_df_out.columns))
    #     self.post_match_df.cache()

    #     return treatment_df_out, control_df_out

    def score_df(self, df=None):
        if df is None:
            df = self.df

        col = self.probability_col
        scored_df = self.propmod.transform(df)
        scored_df = scored_df.withColumn(col+"_1", F.udf(lambda x: float(x[0]), T.FloatType())(col))
        col = col+"_1"
        treatment_df = scored_df.where(F.col(self.label_col) == 1)
        self.treatment_count = treatment_df.count()
        control_can_df = scored_df.where(F.col(self.label_col) == 0)
        return treatment_df, control_can_df

    def make_match_col(self, t_df, c_df, grain):
        col = self.probability_col + "_1"

        probs = list(np.arange(0, 1, self.grain))
        probs = [float(x) for x in probs]

        quantiles = t_df.approxQuantile(col=col, probabilities=probs, relativeError=.05)

        def make_udf(quantiles):
            quantiles = np.array(quantiles)
            group_dict = dict()
            for num in np.arange(0, 1.01, .01):
                num2 = int(round(num*100, 0))
                group_dict[num2] = (num > quantiles).sum()

            def udf(number):
                return int(group_dict.get(int(round(number*100, 0)), -1))
            return udf

        match_col_udf = F.udf(make_udf(quantiles), T.IntegerType())

        t_df = t_df.withColumn(col+"_round", F.round(t_df[col], 2))
        c_df = c_df.withColumn(col+"_round", F.round(c_df[col], 2))

        t_df = t_df.withColumn('match_col', match_col_udf(col+"_round"))
        c_df = c_df.withColumn('match_col', match_col_udf(col+"_round"))
        return t_df, c_df

    def calc_sample_fracs(self, t_df, c_df):
        t_counts = t_df.groupby(['match_col']).count().withColumnRenamed('count', 'treatment')
        c_counts = c_df.groupby(['match_col']).count().withColumnRenamed('count', 'control')
        fracs = t_counts.join(c_counts, on=['match_col'])
        fracs = fracs.toPandas()
        sample_fracs, scale, drop = self.create_util_grid(fracs)
        return sample_fracs[['match_col','treatment_scaled_sample_fraction']],\
               sample_fracs[['match_col','control_scaled_sample_fraction']],\
               scale, drop

    def create_util_grid(self, fracs):
        fracs = fracs.copy(deep=True)
        fracs['control_sample_fraction_naive'] = fracs['treatment']/fracs['control']
        scale_factor = fracs.control_sample_fraction_naive.max()**-1
        if scale_factor >= 1:
            fracs['control_scaled_sample_fraction'] = fracs['control_sample_fraction_naive']
            fracs['treatment_scaled_sample_fraction'] = 1
            fracs = fracs[['match_col', 'treatment_scaled_sample_fraction', 'control_scaled_sample_fraction']]
            return fracs

        scales = np.linspace(1, scale_factor, num=100, endpoint=True)
        options = pd.DataFrame(columns=['scale', 'percent_dropped', 'number'])

        for scale in scales:
            # calc new frac samples
            fracs['control_scaled_sample_fraction'] = np.min(
                [(fracs['treatment'] * scale/fracs['control']).values, [1]*len(fracs)], axis=0)
            fracs['treatment_scaled_sample_fraction'] = fracs['control_scaled_sample_fraction'] * \
                fracs['control']/fracs['treatment']
            #calc %drop
            num_dropped = (fracs['treatment'] * (([scale] * len(fracs)) -
                                                 fracs['treatment_scaled_sample_fraction'])).sum()
            percent_dropped = num_dropped/(fracs['treatment'] * scale).sum()
            # calc new total
            number = (fracs['treatment']*fracs['treatment_scaled_sample_fraction']).sum()
            options = options.append({'scale': scale, 'percent_dropped': percent_dropped,
                                      'number': number}, ignore_index=True)

        # calc util
        options['utility'] = options.apply(self.calc_util_wrapper, axis=1)
        # pick best
        max_util = options.utility.max()
        best_row = options[options.utility == max_util]
        if len(best_row) > 1:
            best_row = best_row.iloc[0]
        winning_scale = best_row['scale'][0]
        winning_drop = best_row['percent_dropped'][0]

        fracs['control_scaled_sample_fraction'] = np.min(
            [(fracs['treatment'] * winning_scale/fracs['control']).values, [1]*len(fracs)], axis=0)
        fracs['treatment_scaled_sample_fraction'] = fracs['control_scaled_sample_fraction'] * \
            fracs['control']/fracs['treatment']
        fracs = fracs[['match_col', 'treatment_scaled_sample_fraction', 'control_scaled_sample_fraction']]

        # return fracs
        return fracs, winning_scale, winning_drop

    @classmethod
    def calc_util_wrapper(cls, row):
        return cls.calc_util(row['number'], row['percent_dropped'])

    @classmethod
    def calc_util(cls, number, dropped):
        log_value = math.log10(number/1000 + 1)
        threshold_boost = cls.logistic_function(L=math.log10(number/1000 + 1)/10, x=number, x0=1000) + \
            cls.logistic_function(L=math.log10(number/5000 + 1)/10, x=number, x0=5000) + \
            cls.logistic_function(L=math.log10(number/500000 + 1)/10, x=number, x0=500000)
        dropped_penalty = 1-min(math.exp(dropped)-1, 1)
        utility = dropped_penalty * (log_value + threshold_boost)
        return utility

    @staticmethod
    def logistic_function(x, L, k=1, x0=0):
        try:
            return L/(1 + math.exp(-k*(x-x0)))
        except OverflowError:
            if x < x0:
                return 0
            if x >= x0:
                return L

    def sample_dfs(self, t_df: pyspark.sql.DataFrame, t_fracs: pd.DataFrame,\
                         c_df: pyspark.sql.DataFrame, c_fracs:pd.DataFrame):
            t_fracs = t_fracs.set_index('match_col').treatment_scaled_sample_fraction.to_dict()
            t_dict = {}
            for key,value in t_fracs.items():
                t_dict[key] = min(float(value),1)
            t_out = t_df.sampleBy(col='match_col', fractions=t_dict, seed=42)

            c_fracs = c_fracs.set_index('match_col').control_scaled_sample_fraction.to_dict()
            c_dict = {}
            for key,value in c_fracs.items():
                c_dict[key] = float(value)
            c_out = c_df.sampleBy(col='match_col', fractions=c_dict, seed=42)
            return t_out, c_out

    def match(self, grain, t_in, c_in):
        t_df, c_df = self.make_match_col(t_in, c_in, grain)
        t_fracs, c_fracs, scaled, dropped = self.calc_sample_fracs(t_df, c_df)
        t_out, c_out = self.sample_dfs(t_df, t_fracs, c_df, c_fracs)
        t_out.cache()
        c_out.cache()
        bias_df, total_bias_reduced = self.eval_match_performance((self.df, t_out.union(c_out)))
        return t_out, c_out, bias_df, total_bias_reduced, dropped, scaled

    def transform(self):
        t_df, c_can_df = self.score_df()
        treatment_df_out, control_df_out, bias_df, total_bias_reduced, dropped, scaled = self.match(
            self.grain, t_df, c_can_df)
        self.matched_treatment = treatment_df_out
        self.matched_control = control_df_out
        self.post_match_df = treatment_df_out.union(control_df_out.select(treatment_df_out.columns))
        self.post_match_df.cache()
        return treatment_df_out, control_df_out

    def impact(self, df=None):
        if df is None:
            df = self.matched_treatment.union(self.matched_control.select(self.matched_treatment.columns))
        # if df.count() < 1000:
        #     return 0, 0, 0

        naive_response_dict = dict()
        response_list = df.groupby(self.label_col).mean(self.response_col).collect()
        naive_response_dict[response_list[0][self.label_col]
                            ] = response_list[0]["avg({col})".format(col=self.response_col)]
        naive_response_dict[response_list[1][self.label_col]
                            ] = response_list[1]["avg({col})".format(col=self.response_col)]

        treatment_rate, control_rate = naive_response_dict[1], naive_response_dict[0]
        if df.count() < 4000:
            return treatment_rate, control_rate, control_rate-treatment_rate

        num_preds = int(df.count()/30)-1
        if num_preds < len(self.pred_cols):
            coeffs = self.propmod.coefficients
            pred_cols = self.pred_cols
            weights = sorted(zip(pred_cols, coeffs), key=lambda x: -abs(x[1]))
            weights = weights[0:num_preds]
            pred_cols = [x[0] for x in weights]
        else:

            pred_cols = self.pred_cols
        # adjusted

        pred_cols_r = self.pred_cols + [self.label_col]
        assembler_r = mlf.VectorAssembler(inputCols=pred_cols_r, outputCol='features_r')
        df = assembler_r.transform(df)
        df.cache()
        lr_r = mlc.LogisticRegression(featuresCol='features_r', labelCol=self.response_col,
                    predictionCol='prediction_{0}'.format(
                    self.response_col), rawPredictionCol='rawPrediction_{0}'.format(self.response_col),
                    probabilityCol='probability_{0}'.format(self.response_col))
        lrm_r = lr_r.fit(df)

        coeff_dict = dict(zip(pred_cols_r, lrm_r.coefficients))

        adjusted_response = control_rate * (1 - math.exp(coeff_dict[self.label_col]))
        return treatment_rate, control_rate, adjusted_response

    def eval_propensity_model(self, by_col_group=None):
        # model metrics
        #   #train always available
        train_metrics_summary = self.propmod.summary
        train_metrics = self.eval_propensity_model_power(train_metrics_summary)
        self.propmod_train_metrics = train_metrics

        #   test only available when object has test_df
        if self.test_set is not None:
            test_metrics_summary = self.propmod.evaluate(self.test_set)
            test_metrics = self.eval_propensity_model_power(test_metrics_summary)
        else:
            test_metrics_summary = None
            warnings.warn("Warning: Did not find test df for propensity model, metrics returned as None")
        self.propmod_test_metrics = test_metrics

        if by_col_group is None:
            by_col_group = False
        model_weights = self.eval_propensity_model_weights(by_col_group)
        return train_metrics, test_metrics, model_weights

    @staticmethod
    def eval_propensity_model_power(summary):
        out = {}
        out['auc'] = summary.areaUnderROC
        out['informativeness'] = summary.fMeasureByThreshold.groupby().max('F-Measure').collect()[0][0]
        out['threshold'] = summary.fMeasureByThreshold.where(
            F.col('F-Measure') == out['informativeness']).collect()[0].threshold
        out['precision'] = summary.precisionByThreshold.where(
            F.col('threshold') == out['threshold']).collect()[0].precision
        out['recall'] = summary.recallByThreshold.where(F.col('threshold') == out['threshold']).collect()[0].recall
        return out

    def eval_propensity_model_weights(self, by_col_group):
        coeffs = self.propmod.coefficients
        pred_cols = self.pred_cols

        if self.pred_cols is not None:
            pred_cols = self.pred_cols
        else:
            warnings.warn("Warning: features columns not provided, using generic names")
            if by_col_group:
                warnings.warn("Warning: without feature columns, cannot group features. Grouping turned off")
                by_col_group = False
            pred_cols = ["c_"]

        # add  model weights to self so they are accesible later even if by_col_group is true
        weights = dict(zip(pred_cols, coeffs))
        self.propmod_weights = weights

        if not by_col_group:
            return weights
        else:
            zipped = zip(pred_cols, coeffs)
            weights_grouped = {}
            for tup in zipped:
                source = tup[0].split('__')[0]
                intermediary = weights_grouped.get(source, 0)
                weights_grouped[source] = intermediary + abs(tup[1])
            self.propmod_weights_grouped = weights_grouped
            return weights_grouped

    def eval_match_performance(self, df_tuple=None):
        assert self.pred_cols is not None, "match performance evaluation requires feature columns"

        if df_tuple is None:
            pre_df = self.df
            post_df = self.post_match_df
        else:
            pre_df = df_tuple[0]
            post_df = df_tuple[1]

        assert set(self.pred_cols).issubset(set(pre_df.columns)) & set(self.pred_cols).issubset(
            set(post_df.columns)), "dataframe must contain feature columns"
        assert (self.label_col in pre_df.columns) & (
            self.label_col in post_df.columns),\
            "dataframes must contain {label} for match performance evaluation".format(label=self.label_col)
        bias_df = self.calc_bias_reduced((pre_df, post_df))
        s_bias_df, total_bias_reduced = self.calc_standard_bias_reduced((pre_df, post_df))

        bias_df = bias_df.join(s_bias_df)
        if df_tuple is not None:
            self.match_performance = bias_df
            self.total_bias_reduced = total_bias_reduced
        return bias_df, total_bias_reduced

    def _calc_bias(self, df):
        bias_df = df.select(self.pred_cols + [self.label_col]).groupby(self.label_col).mean().toPandas().transpose()
        bias_df['bias'] = bias_df[1] - bias_df[0]
        bias_df = bias_df.reset_index()
        bias_df['index'] = bias_df['index'].str.replace(r'avg\(', '').str.replace(r')', '')
        bias_df = bias_df.set_index('index')['bias']
        return bias_df

    def _calc_standard_bias(self, df):
        var_df = self._calc_var(df)
        bias_df = self._calc_bias(df)

        s_bias_df = var_df.join(bias_df)
        s_bias_df['denominator'] = np.sqrt((s_bias_df['var_1'] + s_bias_df['var_0'])/2)
        s_bias_df['standard_bias'] = s_bias_df['bias']/s_bias_df['denominator'] * 100
        s_bias_df = s_bias_df['standard_bias']
        return s_bias_df

    def _calc_var(self, df):
        assert self.pred_cols is not None, "Feature columns must be provided to calculate bias"
        s_var_df = df.groupby(self.label_col).agg({x: 'variance' for x in self.pred_cols}).toPandas().transpose()
        s_var_df = s_var_df.reset_index()
        s_var_df['index'] = s_var_df['index'].str.replace(r')', '').str.replace(r'variance\(', '')
        s_var_df = s_var_df.set_index('index')
        s_var_df.columns = ["var_{0}".format(x) for x in s_var_df.columns]
        return s_var_df

    def calc_bias_reduced(self, df_tuple=None):
        if df_tuple is None:
            pre_df = self.df
            post_df = self.post_match_df
        else:
            pre_df = df_tuple[0]
            post_df = df_tuple[1]

        pre_bias = self._calc_bias(pre_df)
        pre_bias.name = 'pre_bias'

        post_bias = self._calc_bias(post_df)
        post_bias.name = 'post_bias'

        bias_df = pd.concat([pre_bias, post_bias], axis=1)
        bias_df['bias_reduced_absolute'] = bias_df['pre_bias'] - bias_df['post_bias']
        bias_df['bias_reduced_relative'] = bias_df['bias_reduced_absolute']/bias_df['pre_bias'] * 100
        return bias_df

    def calc_standard_bias_reduced(self, df_tuple=None):
        if df_tuple is None:
            pre_df = self.df
            post_df = self.post_match_df
        else:
            pre_df = df_tuple[0]
            post_df = df_tuple[1]

        pre_standard_bias = self._calc_standard_bias(pre_df)
        pre_standard_bias.name = 'pre_standard_bias'

        post_standard_bias = self._calc_standard_bias(post_df)
        post_standard_bias.name = 'post_standard_bias'

        sb_red_df = pd.concat([pre_standard_bias, post_standard_bias], axis=1)
        sb_red_df['standard_bias_reduced_absolute'] = sb_red_df['pre_standard_bias'] - sb_red_df['post_standard_bias']
        sb_red_df['standard_bias_reduced_relative'] = sb_red_df['standard_bias_reduced_absolute'] / \
            sb_red_df['pre_standard_bias']*100

        total_bias_reduced = (1 - sb_red_df.post_standard_bias.abs().sum()/sb_red_df.pre_standard_bias.abs().sum())*100
        return sb_red_df, total_bias_reduced
