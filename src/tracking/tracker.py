"""Log experiments to MLflow and register the best model."""
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "automl-experiments"
REGISTERED_NAME = "automl-best-model"

# Types that appear in models trained by this project (we created them ourselves)
TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


def _log_model(pipe, registered_name):
    """Log the model. Handles both MLflow 3 (`name`) and MLflow 2 (`artifact_path`)."""
    kwargs = dict(registered_model_name=registered_name, skops_trusted_types=TRUSTED_TYPES)
    try:
        mlflow.sklearn.log_model(pipe, name="model", **kwargs)
    except TypeError:
        mlflow.sklearn.log_model(pipe, artifact_path="model", **kwargs)



        
def track_experiment(results: pd.DataFrame, best_pipe, task: str, dataset_name: str,
                     meta: dict, register: bool = True) -> dict:
    """Log one parent run plus one child run per model. Returns run info."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_name = results.loc[results["best"], "model"].iloc[0]
    metric_cols = [c for c in results.columns if c not in ("model", "status", "best")]

    with mlflow.start_run(run_name=f"{dataset_name}-{task}") as parent:
        mlflow.set_tags({"dataset": dataset_name, "task": task, "best_model": best_name})
        mlflow.log_params({k: v for k, v in meta.items() if v is not None})

        for _, row in results.iterrows():
            if row["status"] != "success":
                continue
            with mlflow.start_run(run_name=row["model"], nested=True):
                mlflow.set_tags({"model": row["model"], "task": task, "is_best": str(bool(row["best"]))})
                for col in metric_cols:
                    if pd.notna(row[col]):
                        mlflow.log_metric(col, float(row[col]))
                if row["model"] == best_name:
                    _log_model(best_pipe, REGISTERED_NAME if register else None)

        for col in metric_cols:  # headline metrics on the parent run
            val = results.loc[results["best"], col].iloc[0]
            if pd.notna(val):
                mlflow.log_metric(f"best_{col}", float(val))
        run_id = parent.info.run_id

    version = None
    if register:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
        version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(REGISTERED_NAME, "champion", str(version))

    return {"run_id": run_id, "best_model": best_name, "registered_version": version}


def load_champion():
    """Load the current champion model from the registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")