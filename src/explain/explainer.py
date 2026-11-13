"""Explain a fitted pipeline with SHAP and plain-English summaries."""
import warnings

import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _split_pipeline(pipe):
    """Return (preprocessing steps, final model)."""
    model = pipe.steps[-1][1]
    prep = pipe[:-1]
    return prep, model


def explain_model(pipe, X: pd.DataFrame, sample_size: int = 200, out_dir: str = "../reports"):
    """Compute SHAP values on a sample and save a summary plot.

    Returns {"importance": DataFrame, "summary_plot": path, "text": str}.
    """
    sample = X.sample(min(sample_size, len(X)), random_state=42)
    prep, model = _split_pipeline(pipe)
    Xt = prep.transform(sample)
    if not isinstance(Xt, pd.DataFrame):
        Xt = pd.DataFrame(Xt, columns=[f"f{i}" for i in range(Xt.shape[1])])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(Xt)
        except Exception:
            background = shap.sample(Xt, 50, random_state=42)
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict, background
            )
            values = explainer.shap_values(Xt.head(50), silent=True)
            Xt = Xt.head(50)

    # Normalise to a 2D array of shape (rows, features)
    if isinstance(values, list):
        values = np.mean([np.abs(v) for v in values], axis=0)
    values = np.asarray(values)
    if values.ndim == 3:
        values = np.abs(values).mean(axis=2)

    importance = (
        pd.DataFrame({"feature": Xt.columns, "importance": np.abs(values).mean(axis=0)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance["share_pct"] = (importance["importance"] / importance["importance"].sum() * 100).round(1)

    # Bar chart
    top = importance.head(10).iloc[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh(top["feature"], top["importance"], color="#2563eb")
    plt.xlabel("Mean |SHAP value| (impact on prediction)")
    plt.title("Top features driving the model")
    plt.tight_layout()
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "shap_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return {"importance": importance, "summary_plot": path, "text": summarise(importance)}


def summarise(importance: pd.DataFrame, top_n: int = 3) -> str:
    """Plain-English explanation of the top drivers."""
    top = importance.head(top_n)
    parts = [f"'{r.feature}' ({r.share_pct}%)" for r in top.itertuples()]
    total = round(float(top["share_pct"].sum()), 1)
    return (f"The model's predictions are driven mainly by {', '.join(parts)}. "
            f"Together these account for about {total}% of the total feature impact.")


def explain_why_won(results: pd.DataFrame, task: str) -> str:
    """Plain-English reason the best model won, from the CV results."""
    metric = "f1" if task == "classification" else "r2"
    best = results.iloc[0]
    if len(results) < 2:
        return f"{best['model']} was the only model evaluated."
    second = results.iloc[1]
    gap = round(float(best[metric] - second[metric]), 4)
    std_col = f"{metric}_std"
    std = float(best[std_col]) if std_col in results.columns else None
    msg = (f"{best['model']} won with {metric.upper()} = {best[metric]:.4f}, "
           f"ahead of {second['model']} ({second[metric]:.4f}) by {gap}.")
    if std is not None and gap < std:
        msg += (f" The gap is smaller than the fold-to-fold variation ({std:.4f}), "
                "so the top models are statistically close; prefer the simpler or faster one if needed.")
    return msg