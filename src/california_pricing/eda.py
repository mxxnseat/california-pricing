import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from california_pricing.model.make_model import log_transformer, pipeline
from california_pricing.model.preprocess import TARGET_COLUMN, add_rations

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

def spot_nans(df: pd.DataFrame):
    print(df.isna().sum())

def spot_nan_rows(df: pd.DataFrame):
    print(df.isna().shape)
    nans_rows = df[df.isna().any(axis=1)]
    print(nans_rows.head(5))
    print(nans_rows.tail(5))

def spot_colinear_features(df: pd.DataFrame, save_to = None):
    if "ocean_proximity" in df.columns:
        df = df.drop(columns=["ocean_proximity"])

    corr = df.corr().round(3)
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr, annot=True, center=0, cmap="gnuplot")
    plt.tick_params(axis="x", rotation=90)
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    plt.show()

def spot_skewness(df: pd.DataFrame):
    for column in df:
        sns.violinplot(df, x=column)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="./data/housing.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    df = df.drop(columns=[TARGET_COLUMN])
    df = df.sort_index(axis=1)
    spot_skewness(df)
    spot_nan_rows(df)
    spot_nans(df)
    colinear_pipe = pipeline(DummyRegressor, False)[:-1].set_output(transform="pandas")
    nocolinear_pipe = pipeline(DummyRegressor, True)[:-1].set_output(transform="pandas")

    spot_colinear_features(colinear_pipe.fit_transform(df.copy()), "./figures/colin_log.jpg")
    spot_colinear_features(df.copy(), "./figures/colin_raw.jpg")
    spot_colinear_features(nocolinear_pipe.fit_transform(df.copy()), "./figures/colin_ratios.jpg")