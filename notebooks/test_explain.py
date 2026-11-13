import sys
sys.path.append("..")
from sklearn.datasets import load_iris
from src.profiling.profiler import profile_dataset
from src.profiling.task_detector import detect_task
from src.preprocessing.builder import build_preprocessor
from src.training.trainer import run_experiments
from src.explain.explainer import explain_model, explain_why_won

df = load_iris(as_frame=True).frame
target = "target"
types = profile_dataset(df)["column_types"]
info = detect_task(df, target)
pre, _ = build_preprocessor(df, target, types)
results, best, le = run_experiments(df, target, info["task"], pre)

print(explain_why_won(results, info["task"]))
out = explain_model(best, df.drop(columns=[target]))
print(out["importance"].round(4))
print(out["text"])
print("Plot saved to:", out["summary_plot"])

