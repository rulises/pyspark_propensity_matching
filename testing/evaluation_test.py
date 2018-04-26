from . import Model, Estimator

import pyspark
import pyspark.ml.classification as mlc

import datetime as dt

import pytest

pytestmark = pytest.mark.usefixtures("spark_context", "hive_context", "spark_session", "df")

#auto
#argpass
#fitpass


#model
#   power
#       w/test_df
#       w/o test_df
#   weights
#       w/pred_cols
#           by_group
#           not by group 
#       w/o pred_cols
#           by_group
#           not by group

#matching

