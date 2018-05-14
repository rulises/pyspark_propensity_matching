# flake8: noqa E302
import pytest

from pyspark.sql import SparkSession

import pyspark.ml.feature as mlf

@pytest.fixture(scope="session")
def spark(request):
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    return spark


@pytest.fixture(scope="session")
def df(spark):
    df = spark.read.parquet('df')
    pred_cols = [x for x in df.columns if x not in ['features', 'label', 'response']]
    assembler = mlf.VectorAssembler(inputCols=pred_cols, outputCol='features')
    df = assembler.transform(df)  # type pyspark.sql.DataFrame
    df.cache()
    return df

@pytest.fixture(scope="session")
def pred_cols(df):
    return [x for x in df.columns if x not in ['features', 'label', 'response']]

@pytest.fixture(scope="session")
def feature_col():
    return 'features'

@pytest.fixture(scope="session")
def label_col():
    return 'label'

@pytest.fixture(scope="session")
def response_col():
    return 'response'
