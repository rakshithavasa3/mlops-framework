"""Tune hyperparameters of a model with Optuna."""
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_model(name, task, trial):
    """Build a model with hyperparameters sampled by Optuna."""
    clf = task == "classification"
    if name == "Random Forest":
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            n_jobs=-1, random_state=42,
        )
        return RandomForestClassifier(class_weight="balanced", **p) if clf else RandomForestRegressor(**p)
    if name == "HistGradientBoosting":
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 400),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
            random_state=42,
        )
        return HistGradientBoostingClassifier(**p) if clf else HistGradientBoostingRegressor(**p)
    if name == "XGBoost":
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            n_jobs=-1, random_state=42,
        )
        return XGBClassifier(eval_metric="logloss", **p) if clf else XGBRegressor(**p)
    if name == "Decision Tree":
        p = dict(
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            random_state=42,
        )
        return DecisionTreeClassifier(class_weight="balanced", **p) if clf else DecisionTreeRegressor(**p)
    if name == "Logistic Regression":
        return LogisticRegression(C=trial.suggest_float("C", 1e-3, 100, log=True),
                                  max_iter=2000, class_weight="balanced")
    if name == "Ridge Regression":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))
    raise ValueError(f"No search space for {name}")


def tune_model(name, df, target, task, preprocessor, n_trials=15, cv_folds=5):
    """Return (best_pipeline_fitted, best_params, best_score, n_trials_run)."""
    X = df.drop(columns=[target])
    y = df[target]
    mask = y.notna()
    X, y = X[mask], y[mask]

    if task == "classification":
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
        cv, scoring = StratifiedKFold(cv_folds, shuffle=True, random_state=42), "f1_weighted"
    else:
        cv, scoring = KFold(cv_folds, shuffle=True, random_state=42), "r2"

    def build(model):
        steps = [("prep", clone(preprocessor))]
        if name in ("Logistic Regression", "Ridge Regression"):
            steps.append(("scale", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)

    def objective(trial):
        pipe = build(suggest_model(name, task, trial))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_trial = optuna.trial.FixedTrial(study.best_params)
    best_pipe = build(suggest_model(name, task, best_trial))
    best_pipe.fit(X, y)
    return best_pipe, study.best_params, float(study.best_value), len(study.trials)