import math

from typing import Tuple

import pyspark
import pyspark.ml.feature as mlf
import pyspark.ml.classification as mlc
dataframe = pyspark.sql.DataFrame

NO_IMPACT_THRESHOLD_COUNT = 2000
NAIVE_THRESHOLD_COUNT = 4000
SAMPLES_PER_FEATURE = 100


def impact(df: pyspark.sql.DataFrame, label_col: str, response_col: str, pred_cols_coefficients: zip) -> Tuple[float, float, float]:
    count_ = df.count()

    if count_ < NO_IMPACT_THRESHOLD_COUNT:
        return 0, 0, 0

    # can we use accessors to remove the dict entirely? HACK feels ugly
    naive_response_dict = dict()
    response_list = df.groupby(label_col).mean(response_col).collect()
    naive_response_dict[response_list[0][label_col]] = response_list[0]["avg({col})".format(col=response_col)]
    naive_response_dict[response_list[1][label_col]] = response_list[1]["avg({col})".format(col=response_col)]
    treatment_rate, control_rate = naive_response_dict[1], naive_response_dict[0]

    if count_ < NAIVE_THRESHOLD_COUNT:
        return treatment_rate, control_rate, control_rate-treatment_rate

    # choose fewer features if appropriate to prevent overfit. round down
    num_preds = int(df.count()/SAMPLES_PER_FEATURE)-1
    if num_preds < len(list(pred_cols_coefficients)):
        weights = sorted(pred_cols_coefficients, key=lambda x: -abs(x[1]))
        weights = weights[0:num_preds]
        pred_cols = [x[0] for x in weights]
    else:
        pred_cols = [x[0] for x in pred_cols_coefficients]

    pred_cols_r = pred_cols + [label_col]
    assembler_r = mlf.VectorAssembler(inputCols=pred_cols_r, outputCol='features_r')
    df = assembler_r.transform(df)
    df.cache()
    lre_r = mlc.LogisticRegression(featuresCol='features_r',
                                   labelCol=response_col,
                                   predictionCol='prediction_{0}'.format(response_col),
                                   rawPredictionCol='rawPrediction_{0}'.format(response_col),
                                   probabilityCol='probability_{0}'.format(response_col))
    lrm_r = lre_r.fit(df)

    coeff_dict = dict(zip(pred_cols_r, lrm_r.coefficients))

    adjusted_response = control_rate * (1 - math.exp(coeff_dict[label_col]))
    return treatment_rate, control_rate, adjusted_response
