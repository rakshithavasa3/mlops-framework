import sys
sys.path.append("..")
from sklearn.datasets import fetch_california_housing
from src.profiling.profiler import profile_dataset
from src.profiling.task_detector import detect_task
from src.preprocessing.builder import build_preprocessor
from src.training.tuner import tune_model

df = fetch_california_housing(as_frame=True).frame.sample(3000, random_state=1)
target = "MedHouseVal"
types = profile_dataset(df)["column_types"]
info = detect_task(df, target)
pre, _ = build_preprocessor(df, target, types)

pipe, params, score, n = tune_model("HistGradientBoosting", df, target, info["task"], pre, n_trials=10)
print("Trials run:", n)
print("Best CV R2:", round(score, 4), "(untuned was 0.7823)")
print("Best params:", params)