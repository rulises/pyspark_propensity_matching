import sys
import os

import pytest

import pyspark
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.ml.feature as mlf

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

sys.path.append(os.path.abspath('../'))
from propensity_matching import Estimator
import importlib

spark = SparkSession.builder.master("local[2]").getOrCreate()


def get_df(spark: pyspark.sql.SparkSession, size: int=10**5):
    # gen numpy array
    args = {
        "n_samples": size,
        "n_features": 20,
        "n_informative": 10,
        "n_redundant": 8,
        "n_repeated": 2,
        "n_classes": 2,
        #"n_clusters_per_class" :  ,
        "weights": [.95, .05],
        "flip_y": 0,
        #"class_sep":  ,
        #"hypercube":  ,
        #"shift":  ,
        #"scale":  ,
        #"shuffle":  ,
        "random_state": 42
    }
    data, labels = make_classification(**args)
    # gen pandas
    cols = ["{0}__f".format(str(x)) for x in range(len(data[0]))]
    pd_df = pd.DataFrame(data, columns=cols)  # type pandas.DataFrame
    pd_df['label'] = labels
    # add response
    treatment_prob, control_prob = .1, .2
    pd_df['response'] = None
    pd_df.loc[pd_df.label == 0, 'response'] = np.random.binomial(
        n=1,
        p=control_prob,
        size=pd_df.label.count()-pd_df.label.sum()
    )
    pd_df.loc[pd_df.label == 1, 'response'] = np.random.binomial(
        n=1,
        p=treatment_prob,
        size=pd_df.label.sum()
    )
    # gen sparkdf
    spark_df = spark.createDataFrame(pd_df)  # type pyspark.sql.DataFrame
    spark_df = spark_df.withColumn('0__e', pyspark.sql.functions.lit(0))
    cols += ['0__e']
    assembler = mlf.VectorAssembler(inputCols=cols, outputCol='features')
    spark_df = assembler.transform(spark_df)  # type pyspark.sql.DataFrame
    spark_df.cache()
    return spark_df

df = get_df(spark)

non_pred_cols = ['label', 'response', 'features']
pred_cols = [x for x in df.columns if x not in non_pred_cols]

import propensity_matching
importlib.reload(propensity_matching)
# importlib.reload(Estimator)

estimator = propensity_matching.Estimator.Estimator(pred_cols)
model = estimator.fit(df)

t,c = model.transform(df)

treatment_rate, control_rate, adjusted_response = model.impact(t,c)

print("""
treatment rate: {treatment_rate}
control_rate: {control_rate}
adjusted_response: {adjusted_response}
""".format(treatment_rate=treatment_rate, control_rate=control_rate, adjusted_response=adjusted_response))

post_df = t.union(c.select(t.columns))
eval_out = model.evaluate(df, post_df, post_df, True)