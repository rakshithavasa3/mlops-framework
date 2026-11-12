"""Rank candidate algorithms from dataset meta-features, with reasons."""
import numpy as np
import pandas as pd


def compute_meta_features(df, target, task, types, imbalance_ratio=None):
    """Summarise the dataset in a few numbers the recommender can reason about."""
    X = df.drop(columns=[target])
    n_rows, n_cols = X.shape
    num = [c for c in types["numerical"] if c != target]
    cat = [c for c in types["categorical"] if c != target]
    skews = [abs(float(X[c].skew())) for c in num if X[c].nunique() > 2]
    return {
        "n_rows": int(n_rows),
        "n_features": int(n_cols),
        "numeric_ratio": round(len(num) / max(n_cols, 1), 2),
        "categorical_ratio": round(len(cat) / max(n_cols, 1), 2),
        "missing_ratio": round(float(X.isna().mean().mean()), 3),
        "avg_skew": round(float(np.mean(skews)), 2) if skews else 0.0,
        "imbalance_ratio": imbalance_ratio,
        "n_classes": int(df[target].nunique()) if task == "classification" else None,
    }


def recommend(task: str, meta: dict) -> list:
    """Return a ranked list of {model, score, reasons}."""
    n, f = meta["n_rows"], meta["n_features"]
    small, large = n < 1000, n > 50000
    scores = {}

    def add(model, pts, reason):
        entry = scores.setdefault(model, {"score": 3.0, "reasons": []})
        entry["score"] += pts
        entry["reasons"].append(reason)

    linear = "Logistic Regression" if task == "classification" else "Ridge Regression"

    # Tree ensembles: strong general-purpose choice for tabular data
    for m in ("Random Forest", "HistGradientBoosting", "XGBoost"):
        add(m, 1.0, "Tree ensembles are strong on tabular data.")
    if meta["categorical_ratio"] > 0.3:
        for m in ("Random Forest", "HistGradientBoosting", "XGBoost"):
            add(m, 0.5, "Handles many categorical features well.")
    if meta["avg_skew"] > 1:
        for m in ("Random Forest", "HistGradientBoosting", "XGBoost"):
            add(m, 0.5, "Robust to skewed numeric features.")
    if small:
        add("Random Forest", 0.5, f"Resists overfitting on a small dataset ({n} rows).")
        add("HistGradientBoosting", -0.5, "Boosting can overfit very small datasets.")
        add("XGBoost", -0.5, "Boosting can overfit very small datasets.")
    if n >= 1000:
        add("HistGradientBoosting", 0.5, f"Fast and accurate on {n} rows.")
        add("XGBoost", 0.5, f"Scales well to {n} rows.")
    if large:
        add("Random Forest", -0.5, "Slower on very large datasets.")

    # Linear baseline
    add(linear, 0.5, "A fast, interpretable baseline.")
    if meta["numeric_ratio"] > 0.7:
        add(linear, 0.5, "Dataset is mostly numeric.")
    if f > n:
        add(linear, 0.5, "More features than rows: simpler models generalise better.")

    # Decision tree: simple, explainable
    add("Decision Tree", 0.0, "Simple and easy to explain, but prone to overfitting.")
    if small:
        add("Decision Tree", 0.5, "Works as a quick baseline on small data.")

    # Task-specific notes
    if task == "classification" and meta["imbalance_ratio"] and meta["imbalance_ratio"] > 3:
        for m in ("Random Forest", "Logistic Regression", "Decision Tree"):
            add(m, 0.5, "Supports class weights for the detected class imbalance.")

    ranked = [{"model": m, "score": round(v["score"], 2), "reasons": v["reasons"]} for m, v in scores.items()]
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked