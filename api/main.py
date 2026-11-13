"""FastAPI backend for the AutoML framework."""
import io
import json
import os
import sys

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.explain.explainer import explain_model, explain_why_won
from src.packaging.packager import package_model, predict as run_predict
from src.preprocessing.builder import build_preprocessor
from src.profiling.profiler import profile_dataset
from src.profiling.task_detector import detect_task
from src.tracking.tracker import track_experiment
from src.training.recommender import compute_meta_features, recommend
from src.training.trainer import run_experiments

MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
BUDGETS = {"quick": 3, "standard": 5, "thorough": 5}

app = FastAPI(title="AutoML Framework API", version="0.1")


def read_csv(upload: UploadFile) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(upload.file.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    df = read_csv(file)
    return {"filename": file.filename, "columns": list(df.columns), "profile": profile_dataset(df)}


@app.post("/recommend")
def recommend_endpoint(file: UploadFile = File(...), target: str = Form(...)):
    df = read_csv(file)
    info = detect_task(df, target)
    if not info["valid"]:
        return {"task_info": info, "recommendations": []}
    types = profile_dataset(df)["column_types"]
    meta = compute_meta_features(df, target, info["task"], types, info["imbalance_ratio"])
    return {"task_info": info, "meta_features": meta, "recommendations": recommend(info["task"], meta)}


@app.post("/run")
def run(file: UploadFile = File(...), target: str = Form(...), budget: str = Form("standard")):
    df = read_csv(file)
    name = os.path.splitext(file.filename or "dataset")[0]

    info = detect_task(df, target)
    if not info["valid"]:
        raise HTTPException(422, {"message": "Invalid target", "errors": info["errors"]})

    types = profile_dataset(df)["column_types"]
    meta = compute_meta_features(df, target, info["task"], types, info["imbalance_ratio"])
    pre, notes = build_preprocessor(df, target, types)

    try:
        results, best, le = run_experiments(
            df, target, info["task"], pre, max_models=BUDGETS.get(budget, 5)
        )
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")

    X = df.drop(columns=[target])
    exp = explain_model(best, X, out_dir=REPORTS_DIR)
    why = explain_why_won(results, info["task"])
    tracked = track_experiment(results, best, info["task"], name, meta)
    folder = package_model(best, results, info["task"], name, target, X.columns, le,
                           explanation=exp["text"], out_root=MODELS_DIR)

    return {
        "task_info": info,
        "preprocessing_notes": notes,
        "results": json.loads(results.to_json(orient="records")),
        "why_it_won": why,
        "feature_importance": json.loads(exp["importance"].head(10).to_json(orient="records")),
        "explanation": exp["text"],
        "mlflow": tracked,
        "model_id": os.path.basename(folder),
    }


@app.get("/models")
def list_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(d for d in os.listdir(MODELS_DIR)
                  if os.path.isfile(os.path.join(MODELS_DIR, d, "final_model.joblib")))


@app.post("/predict/{model_id}")
def predict_endpoint(model_id: str, file: UploadFile = File(...)):
    if "/" in model_id or "\\" in model_id or ".." in model_id:
        raise HTTPException(400, "Invalid model id")
    folder = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(folder):
        raise HTTPException(404, "Model not found")
    df = read_csv(file)
    try:
        out = run_predict(folder, df)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return json.loads(out.to_json(orient="records"))