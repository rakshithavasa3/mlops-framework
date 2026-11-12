"""Train and compare candidate models with cross-validation."""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def get_candidates(task: str) -> dict:
    """Return {name: estimator} for the given task."""
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", n_jobs=-1, random_state=42),
        }
    return {
        "Ridge Regression": Ridge(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    }


# Models that need scaled inputs (the preprocessor does not scale date features)
NEEDS_SCALING = {"Logistic Regression", "Ridge Regression"}


def run_experiments(df, target, task, preprocessor, max_models=5, cv_folds=5):
    """Cross-validate each candidate. Returns (results_df, fitted_best_pipeline, label_encoder)."""
    X = df.drop(columns=[target])
    y = df[target]

    # Drop rows with a missing target
    mask = y.notna()
    X, y = X[mask], y[mask]

    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2", "mae": "neg_mean_absolute_error"}

    candidates = dict(list(get_candidates(task).items())[:max_models])
    rows, pipelines = [], {}

    for name, model in candidates.items():
        steps = [("prep", clone(preprocessor))]
        if name in NEEDS_SCALING:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", model))
        pipe = Pipeline(steps)

        start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
            elapsed = time.time() - start

            if task == "classification":
                row = {
                    "model": name, "status": "success",
                    "accuracy": scores["test_accuracy"].mean(),
                    "f1": scores["test_f1"].mean(),
                    "f1_std": scores["test_f1"].std(),
                    "time_s": round(elapsed, 2),
                }
            else:
                row = {
                    "model": name, "status": "success",
                    "rmse": -scores["test_rmse"].mean(),
                    "r2": scores["test_r2"].mean(),
                    "mae": -scores["test_mae"].mean(),
                    "r2_std": scores["test_r2"].std(),
                    "time_s": round(elapsed, 2),
                }
            pipelines[name] = pipe
        except Exception as e:
            row = {"model": name, "status": f"failed: {e}"}
        rows.append(row)

    results = pd.DataFrame(rows)
    ok = results[results["status"] == "success"]
    if ok.empty:
        raise RuntimeError("All models failed. Check the data and target.")

    sort_col, ascending = ("f1", False) if task == "classification" else ("r2", False)
    results = results.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)

    best_name = results.iloc[0]["model"]
    best_pipe = pipelines[best_name]
    best_pipe.fit(X, y)  # refit on all data
    results["best"] = results["model"] == best_name
    return results, best_pipe, label_encoder