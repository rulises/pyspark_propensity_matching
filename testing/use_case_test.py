
import sys
import os
import datetime as dt

import pyspark

import pytest

sys.path.append(os.path.abspath('../'))
from propensity_matching import Estimator
from propensity_matching import Model

import importlib
def reset():
    importlib.reload(Model)
    importlib.reload(Estimator)


pytestmark = pytest.mark.usefixtures("spark", "df")

def test_case(df):

    reset()
    propensity_estimator_args = {
        "pred_cols": [col for col in df.columns if col not in ['label', 'response', 'features']]
    }
    propensity_estimator = Estimator.Estimator(**propensity_estimator_args)
    propensity_model = propensity_estimator.fit(df)
    treatment, control = propensity_model.transform()
    treatment.show()
    control.show()
    impact = propensity_model.impact()
    print(impact)
    match_performance = propensity_model.eval_match_performance()
    match_performance.describe()
    model_power = propensity_model.eval_propensity_model()
    print(model_power)
    return True
