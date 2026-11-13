"""Package the best model and run predictions from the package."""
import json
import os
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn


def package_model(pipe, results, task, dataset_name, target, X_columns,
                  label_encoder=None, explanation=None, out_root="../models"):
    """Save final_model.joblib, metadata.json and metrics.json. Returns the folder path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(out_root, f"{dataset_name}_{stamp}")
    os.makedirs(folder, exist_ok=True)

    joblib.dump({"pipeline": pipe, "label_encoder": label_encoder},
                os.path.join(folder, "final_model.joblib"))

    best = results[results["best"]].iloc[0]
    metadata = {
        "dataset": dataset_name,
        "task": task,
        "target": target,
        "best_model": best["model"],
        "feature_columns": list(X_columns),
        "classes": [str(c) for c in label_encoder.classes_] if label_encoder is not None else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "explanation": explanation,
    }
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    metrics = json.loads(results.to_json(orient="records"))
    with open(os.path.join(folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return folder


def load_package(folder):
    """Load a packaged model. Returns (pipeline, label_encoder, metadata)."""
    bundle = joblib.load(os.path.join(folder, "final_model.joblib"))
    with open(os.path.join(folder, "metadata.json")) as f:
        metadata = json.load(f)
    return bundle["pipeline"], bundle["label_encoder"], metadata


def predict(folder, new_data: pd.DataFrame) -> pd.DataFrame:
    """Predict on new rows. Raises a clear error if columns are missing."""
    pipe, le, meta = load_package(folder)
    missing = [c for c in meta["feature_columns"] if c not in new_data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    X = new_data[meta["feature_columns"]]

    preds = pipe.predict(X)
    out = new_data.copy()
    if le is not None:
        out["prediction"] = le.inverse_transform(preds)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            out["confidence"] = np.round(proba.max(axis=1), 4)
    else:
        out["prediction"] = preds
    return out