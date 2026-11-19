import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from california_pricing.model.preprocess import LOG_COLUMNS, add_rations, preprocess

def get_feature_names(transformer, input_features):
    df = pd.DataFrame(columns=input_features, dtype=float)
    res = add_rations(df).columns.to_numpy()
    return res

def log_transformer():
    return ColumnTransformer(
        transformers=[
            ("log", FunctionTransformer(np.log1p), LOG_COLUMNS)
        ], remainder="passthrough", verbose_feature_names_out=False)

def pipeline(regressor, solve_colin = True):
    if solve_colin:
        return Pipeline([
            ("log", log_transformer().set_output(transform="pandas")),
            ("rations", FunctionTransformer(add_rations, feature_names_out=get_feature_names).set_output(transform="pandas")),
            ("preprocess", preprocess),
            ("reg", regressor)
        ])
    return Pipeline([
        ("log", log_transformer().set_output(transform="pandas")),
        ("preprocess", preprocess),
        ("reg", regressor)
    ])

def make_model(regressor, solve_colin = True):
    reg = pipeline(regressor, solve_colin)

    return TransformedTargetRegressor(reg, transformer=StandardScaler())