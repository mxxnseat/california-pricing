from typing import Any
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

def build_stats_table(stats: dict[str, float]):
    return pd.DataFrame(list(stats.items()), columns=["model", "rmse"])

def select_model(models_to_try: dict[str, TransformedTargetRegressor], X, y, cv_args):
    best_name = best_model = None
    best_score = float("+inf")
    stats = {}
    for name, model in models_to_try.items():
        score = -cross_val_score(model, X, y, n_jobs=cv_args["n_jobs"], cv=cv_args["cv"], scoring="neg_root_mean_squared_error")
        score_mean = score.mean()
        stats[name] = score_mean
        if score_mean < best_score:
            best_name = name
            best_model = model
            best_score = score_mean

    return best_name, best_model, best_score, build_stats_table(stats)

def select_best_model_params(searches: dict[str, GridSearchCV[Any]], X, y):
    best_model = None
    best_name = None
    best_params = None
    best_score = float("+inf")
    stats = {}
    for name, search in searches.items():
        search.fit(X, y)
        score = -search.best_score_
        stats[name] = score

        if score < best_score:
            best_score = score
            best_params = search.best_params_
            best_name = name
            best_model = search.best_estimator_

    return best_name, best_model, best_params, build_stats_table(stats)
