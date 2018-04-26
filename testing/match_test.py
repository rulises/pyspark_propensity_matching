#from . import Model, Estimator

import pyspark
import pyspark.ml.classification as mlc

import datetime as dt

import pytest

pytestmark = pytest.mark.usefixtures("spark_context", "hive_context", "spark_session", "df")


import importlib

from propensity_matching import Estimator
from propensity_matching import Model
importlib.reload(Estimator)
importlib.reload(Model)

df = gen_df(spark_session)
propensity_estimator_args = {
    "pred_cols" : [col for col in df.columns if col not in ['label', 'response', 'features']]
    }


propensity_estimator = Estimator.Estimator(**propensity_estimator_args)
propensity_model = propensity_estimator.fit(df)
#treatment, control = propensity_model.transform()
#treatment.show()
#control.show()

t_df, c_c_df = propensity_model.score_df()
t_df, c_c_df = propensity_model.make_match_col(t_df, c_c_df, .1)


fracs['control_sample_fraction_naive'] = fracs['control']/fracs['treatment']
fracs

scale_factor = fracs.control_sample_fraction_naive.max() **-1
original_count = fracs.treatment.sum()

#scale tolerances

#%reduction tolerances

#score_Df

#make_match_col

#calc_sample_fracs

#sample_dfs

#match

#transform