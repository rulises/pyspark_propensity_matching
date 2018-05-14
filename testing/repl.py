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
from propensity_matching import estimator
import importlib

spark = SparkSession.builder.master("local[2]").getOrCreate()

df = spark.read.parquet('df')
pred_cols = [x for x in df.columns if x not in ['features', 'label', 'response']]
assembler = mlf.VectorAssembler(inputCols=pred_cols, outputCol='features')
df = assembler.transform(df)  # type pyspark.sql.DataFrame
df.cache()

non_pred_cols = ['label', 'response', 'features']
pred_cols = [x for x in df.columns if x not in non_pred_cols]

import propensity_matching
importlib.reload(propensity_matching)
# importlib.reload(Estimator)

estimator = propensity_matching.Estimator.Estimator(pred_cols)
model = estimator.fit(df)
df = model.df
t,c = model.transform(df)

treatment_rate, control_rate, adjusted_response = model.impact(t,c)

print("""
treatment rate: {treatment_rate}
control_rate: {control_rate}
adjusted_response: {adjusted_response}
""".format(treatment_rate=treatment_rate, control_rate=control_rate, adjusted_response=adjusted_response))

post_df = t.union(c.select(t.columns))
eval_out = model.evaluate(df, post_df, post_df, True)