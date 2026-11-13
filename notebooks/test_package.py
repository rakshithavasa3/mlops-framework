import sys
sys.path.append("..")
from sklearn.datasets import load_iris
from src.profiling.profiler import profile_dataset
from src.profiling.task_detector import detect_task
from src.preprocessing.builder import build_preprocessor
from src.training.trainer import run_experiments
from src.explain.explainer import explain_model
from src.packaging.packager import package_model, predict

df = load_iris(as_frame=True).frame
target = "target"
types = profile_dataset(df)["column_types"]
info = detect_task(df, target)
pre, _ = build_preprocessor(df, target, types)
results, best, le = run_experiments(df, target, info["task"], pre)
X = df.drop(columns=[target])
exp = explain_model(best, X)

folder = package_model(best, results, info["task"], "iris", target,
                       X.columns, le, explanation=exp["text"])
print("Packaged at:", folder)

print(predict(folder, X.head(3)))
try:
    predict(folder, X.drop(columns=["petal width (cm)"]).head(3))
except ValueError as e:
    print("Good, bad input rejected:", e)