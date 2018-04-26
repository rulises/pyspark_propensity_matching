from . import Model, Estimator

import pyspark
import pyspark.ml.classification as mlc

import datetime as dt

import pytest

pytestmark = pytest.mark.usefixtures("spark_context", "hive_context", "spark_session", "df")


@pytest.fixture
def estimator(scope='session'):
    estimator = Estimator()
    return estimator

@pytest.fixture
def propensity_estimator_args():
    propensity_estimator_args ={
    "featuresCol":"features", 
    "labelCol":"label", 
    "predictionCol":"prediction", 
    "maxIter":10, 
    "regParam":.2, 
    "elasticNetParam":0.5, 
    #tol":1e-6, 
    "fitIntercept":True, 
    #"threshold":0.5, 
    #"thresholds":None, 
    "probabilityCol":"probability", 
    #rawPredictionCol":"rawPrediction", 
    #"standardization":True, 
    #"weightCol":None, 
    #aggregationDepth":2, 
    "family":"binomial"
    }
    return propensity_estimator_args

@pytest.fixture
def model(estimator, df, scope='session'):
    #tests autofit functionality
    try:
        model = estimator.fit(df)
    except Exception as e:
        model = e
    return model

@pytest.fixture
def transformed_model(model, scope='session'):
    try:
        return model.transform()
    except Exception as e:
        return e
# steps:
# 1) load data
    #covered by df fixture
# 2) fit predictor
    # auto fit
def test_auto_fit(model):
    assert(type(model) == Model), str(model)
    # pass fitted
def test_prefit(propensity_estimator_args, df):
    propensity_estimator = mlc.LogisticRegression(**propensity_estimator_args)
    propensity_model = propensity_estimator.fit(df)
    estimator = Estimator(propensity_model=propensity_model)
    model = estimator.fit(df)


    # pass args
def test_argfit(df):
    propensity_estimator_args ={
        "featuresCol":"features", 
        "labelCol":"label", 
        "predictionCol":"prediction", 
        "maxIter":10, 
        "regParam":.2, 
        "elasticNetParam":0.5, 
        #tol":1e-6, 
        "fitIntercept":False, 
        #"threshold":0.5, 
        #"thresholds":None, 
        "probabilityCol":"probability", 
        #rawPredictionCol":"rawPrediction", 
        #"standardization":True, 
        #"weightCol":None, 
        #aggregationDepth":2, 
        "family":"binomial"
        }

    propensity_estimator = mlc.LogisticRegression
    estimator = Estimator(propensity_estimator_args=propensity_estimator_args , 
                          propensity_estimator=propensity_estimator
                          )
    model = estimator.fit(df)


# 3) match
    #normal
def test_normal_match(transformed_model):
    assert type(transformed_model) == Model, str(transformed_model)
    #proportion mismatch 
def test_imbalance_match(model,  df):
    df = df.sampleBy('label', fractions={0:.2, 1:1}, seed=42)
    model.transform(df)
    
# 4) evaluate impact
def test_impact(transformed_model):
    treatment_rate, control_rate, adjusted_response = transformed_model.impact()
    #naive
    #adjusted
# 5) governance
    #bias reduced
    #model stability
    
@pytest.fixture
def estimator(scope='session'):
    estimator = Estimator()
    return estimator


@pytest.fixture
def propensity_estimator_args():
    propensity_estimator_args = {
        "featuresCol": "features",
        "labelCol": "label",
        "predictionCol": "prediction",
        "maxIter": 10,
        "regParam": .2,
        "elasticNetParam": 0.5,
        # tol":1e-6,
        "fitIntercept": True,
        #"threshold":0.5,
        #"thresholds":None,
        "probabilityCol": "probability",
        # rawPredictionCol":"rawPrediction",
        #"standardization":True,
        #"weightCol":None,
        # aggregationDepth":2,
        "family": "binomial"
    }
    return propensity_estimator_args


@pytest.fixture
def model(estimator, df, scope='session'):
    # tests autofit functionality
    try:
        model = estimator.fit(df)
    except Exception as e:
        model = e
    return model


@pytest.fixture
def transformed_model(model, scope='session'):
    try:
        return model.transform()
    except Exception as e:
        return e
# steps:
# 1) load data
    # covered by df fixture
# 2) fit predictor
    # auto fit


def test_auto_fit(model):
    assert(type(model) == Model), str(model)
    # pass fitted


def test_prefit(propensity_estimator_args, df):
    propensity_estimator = mlc.LogisticRegression(**propensity_estimator_args)
    propensity_model = propensity_estimator.fit(df)
    estimator = Estimator(propensity_model=propensity_model)
    model = estimator.fit(df)

    # pass args


def test_argfit(df):
    propensity_estimator_args = {
        "featuresCol": "features",
        "labelCol": "label",
        "predictionCol": "prediction",
        "maxIter": 10,
        "regParam": .2,
        "elasticNetParam": 0.5,
        # tol":1e-6,
        "fitIntercept": False,
        #"threshold":0.5,
        #"thresholds":None,
        "probabilityCol": "probability",
        # rawPredictionCol":"rawPrediction",
        #"standardization":True,
        #"weightCol":None,
        # aggregationDepth":2,
        "family": "binomial"
    }

    propensity_estimator = mlc.LogisticRegression
    estimator = Estimator(propensity_estimator_args=propensity_estimator_args,
                          propensity_estimator=propensity_estimator
                          )
    model = estimator.fit(df)


# 3) match
    # normal
def test_normal_match(transformed_model):
    assert type(transformed_model) == Model, str(transformed_model)
    # proportion mismatch


def test_imbalance_match(model,  df):
    df = df.sampleBy('label', fractions={0: .2, 1: 1}, seed=42)
    model.transform(df)

# 4) evaluate impact


def test_impact(transformed_model):
    treatment_rate, control_rate, adjusted_response = transformed_model.impact()
    # naive
    # adjusted
# 5) governance
    # bias reduced
    # model stability

