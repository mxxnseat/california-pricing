import pandas as pd
from sklearn.compose import ColumnTransformer,  make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "median_house_value"
CATEGORICAL_FEATURES = ["ocean_proximity"]
LOG_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households"]

num_pipeline = Pipeline([
    ("imputer",  SimpleImputer()),
    ("scaler", StandardScaler())]
)
categorical_pipeline = Pipeline([("onehot", OneHotEncoder(drop="first", sparse_output=False))])

preprocess = ColumnTransformer([
    ("num", num_pipeline, selector(dtype_include="number")),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
], verbose_feature_names_out=False)

def add_rations(df: pd.DataFrame):
    df = df.copy()
    df["total_rooms_total_bedrooms"] = df["total_rooms"] / df["total_bedrooms"]
    df["total_bedrooms_population"] = df["total_bedrooms"] / df["population"]
    df["population_households"] = df["population"] / df["households"]
    
    return df.drop(columns=["total_rooms", "total_bedrooms", "population", "households"])
