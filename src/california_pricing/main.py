import pandas as p
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,  LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
from california_pricing.model.evaluate import evaluate
from california_pricing.model.make_model import make_model
from california_pricing.model.preprocess import TARGET_COLUMN
from california_pricing.model.selection import select_best_model_params

def compare(data: dict[str, dict[str, float]], figure_prefix = ''):
    dfs = pd.DataFrame(data).T

    metrics = dfs.columns
    _, axes = plt.subplots(1, len(metrics), figsize=(12, 6))
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=dfs, hue=dfs.index, ax=ax, x=dfs.index, y=metric, legend=True)
        ax.set_ylabel(metric)
        ax.tick_params("x", rotation=90, rotation_mode="xtick")
        ax.set_xlabel("")
        
    plt.tight_layout()
    plt.savefig(f"./figures/final_bar_comparison{figure_prefix}.jpg")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    table = pd.plotting.table(ax=ax, data=dfs, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_axis_off()
    plt.savefig(f"./figures/final_table_comparison{figure_prefix}.jpg")
    plt.show()

def main(args):
    df = p.read_csv(args.dataset_path)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "baseline (mean)": make_model(DummyRegressor(strategy="mean")), 
        "Linear": make_model(LinearRegression()),
        "Ridge (l2)": make_model(Ridge()),
        "Lasso (l1)": make_model(Lasso()),
        "Gradient boostring": make_model(GradientBoostingRegressor()),
        "Random Forest": make_model(RandomForestRegressor())
    }

    l_search = GridSearchCV(models["Linear"],param_grid={},
        n_jobs=args.n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=5
    )
    dr_search = GridSearchCV(models["baseline (mean)"],param_grid={},
        n_jobs=args.n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=5
    )
    gb_search = GridSearchCV(models["Gradient boostring"],param_grid={
            "regressor__reg__max_depth": [3, 5, 7, 10],
            "regressor__reg__learning_rate": [1e-2, 1e-1],
            "regressor__reg__n_estimators": [50, 100, 200]
        },
        n_jobs=args.n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=5
    )

    ll1_search = GridSearchCV(models["Lasso (l1)"],param_grid={
            "regressor__reg__alpha": [1e-3, 1e-2, 1e-1, 1.0],
        },
        n_jobs=args.n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=5
    )

    ll2_search = GridSearchCV(models["Ridge (l2)"],param_grid={
            "regressor__reg__alpha": [1e-3, 1e-2, 1e-1, 1.0],
        },
        n_jobs=args.n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=5
    )

    rf_search = GridSearchCV(models["Random Forest"],param_grid={
            "regressor__reg__max_depth": [3, 5, 7, 10],
            "regressor__reg__n_estimators": [50, 100, 200]
        },
        n_jobs=args.n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=5
    )

    searches = {
        "baseline (mean)": dr_search, 
        "Gradient Boosting": gb_search, 
        "Random Forest": rf_search, 
        "Lasso (l1)": ll1_search,  
        "Ridge (l2)": ll2_search,
        "Linear": l_search
    }
    
    best_name, best_model, best_params, train_stats = select_best_model_params(searches, X_train, y_train)
    if best_model is None or best_name is None or best_params is None:
        return
    print(f"Best model is {best_name} and it is scored {evaluate(best_model, X_test, y_test)} with params {best_params}")

    test_stats = {}
    for name, model in searches.items():
        test_stats[name] = evaluate(model, X_test, y_test, ["rmse", "mae", "r2"])
    return test_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="./data/housing.csv")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()
    test_stats = main(args)
    assert test_stats is not None
    compare(test_stats, "")
