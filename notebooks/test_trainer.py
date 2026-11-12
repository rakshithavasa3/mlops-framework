import sys
sys.path.append("..")
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from src.profiling.profiler import profile_dataset
from src.profiling.task_detector import detect_task
from src.preprocessing.builder import build_preprocessor
from src.training.trainer import run_experiments

pd.set_option("display.width", 200)


def run(df, target):
    types = profile_dataset(df)["column_types"]
    info = detect_task(df, target)
    print("Task:", info["task"], "| valid:", info["valid"])
    if not info["valid"]:
        print("Errors:", info["errors"])
        return
    pre, notes = build_preprocessor(df, target, types)
    results, best, le = run_experiments(df, target, info["task"], pre)
    print(results.round(4).to_string())
    print()


print("=== IRIS (classification) ===")
run(load_iris(as_frame=True).frame, "target")

print("=== CALIFORNIA HOUSING (regression, 3000-row sample) ===")
run(fetch_california_housing(as_frame=True).frame.sample(3000, random_state=1), "MedHouseVal")