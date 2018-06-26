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
import importlib

spark = SparkSession.builder.master("local[2]").getOrCreate()
df = spark.read.parquet('testing/df')
non_pred_cols = ['label', 'response', 'features']
pred_cols = [x for x in df.columns if x not in non_pred_cols]
assembler = mlf.VectorAssembler(inputCols=pred_cols, outputCol='features')
df = assembler.transform(df)
df.cache()


sys.path.append(os.path.abspath('../'))
import importlib
import propensity_matching



importlib.reload(propensity_matching)
from propensity_matching.estimator import PropensityEstimator
estimator = PropensityEstimator()
model, df2 = estimator.fit(df)
df3, match_info = model.transform(df2)
treatment_rate, control_rate, adjusted_response = model.determine_impact(df3)
perf_sum = model.evaluate_performance(pre_df=df2, post_df=df3, transform_df=df2)

