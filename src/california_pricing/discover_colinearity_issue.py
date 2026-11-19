import argparse

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from california_pricing.model.make_model import make_model
from california_pricing.model.preprocess import TARGET_COLUMN

def main(args):
    df = pd.read_csv(args.dataset_path)

    features_to_explore = ["total_rooms", "population"]
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    results = {
        "linear_l1_colin": [],
        "linear_l1_no_colin": [],
        "linear_colin": [],
        "linear_no_colin": [],
        "gradient_boosting_colin": [],
        "gradient_boosting_no_colin": [],
    }

    n = 20

    for seed in range(n):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        rng = np.random.default_rng(seed)

        X_test_broken = X_test.copy()
        X_test_broken["total_rooms"] = rng.permutation(
            X_test_broken["total_rooms"].to_numpy()
        )

        print(f"\nSeed {seed}")
        print("Original correlation:")
        print(X_test[features_to_explore].corr().round(3))
        print("Modified correlation:")
        print(X_test_broken[features_to_explore].corr().round(3))

        models = {
            "linear_l1_colin": make_model(Lasso(0.1), False),
            "linear_l1_no_colin": make_model(Lasso(0.1), True),
            "linear_colin": make_model(LinearRegression(), False),
            "linear_no_colin": make_model(LinearRegression(), True),
            "gradient_boosting_colin": make_model(
                GradientBoostingRegressor(random_state=seed), False
            ),
            "gradient_boosting_no_colin": make_model(
                GradientBoostingRegressor(random_state=seed), True
            ),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

            train_rmse = root_mean_squared_error(
                y_train, model.predict(X_train)
            )

            original_rmse = root_mean_squared_error(
                y_test, model.predict(X_test)
            )

            broken_rmse = root_mean_squared_error(
                y_test, model.predict(X_test_broken)
            )

            delta = broken_rmse - original_rmse

            results[name].append({
                "train": train_rmse,
                "original": original_rmse,
                "broken": broken_rmse,
                "delta": delta,
            })

    rows = []

    for model_name, scores in results.items():
        for seed, score in enumerate(scores):
            rows.append({
                "model": model_name,
                "seed": seed,
                "train_rmse": score["train"],
                "original_rmse": score["original"],
                "broken_rmse": score["broken"],
                "delta_rmse": score["delta"],
            })

    results_df = pd.DataFrame(rows)

    summary = results_df.groupby("model").agg(
        train_rmse_mean=("train_rmse", "mean"),
        train_rmse_std=("train_rmse", "std"),
        original_rmse_mean=("original_rmse", "mean"),
        original_rmse_std=("original_rmse", "std"),
        broken_rmse_mean=("broken_rmse", "mean"),
        broken_rmse_std=("broken_rmse", "std"),
        delta_rmse_mean=("delta_rmse", "mean"),
        delta_rmse_std=("delta_rmse", "std"),
    )

    plot_df = summary[
        ["original_rmse_mean", "broken_rmse_mean"]
    ].rename(columns={
        "original_rmse_mean": "Original test",
        "broken_rmse_mean": "Broken test",
    })

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.75,
        ylabel="RMSE",
        title="Model performance before and after feature corruption",
        rot=20,
    )

    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("./figures/colinearity_issue.jpg", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="./data/housing.csv")
    args = parser.parse_args()
    main(args)