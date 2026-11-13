import sys
sys.path.append("..")
from sklearn.datasets import load_iris
from src.profiling.profiler import profile_dataset
from src.profiling.task_detector import detect_task
from src.preprocessing.builder import build_preprocessor
from src.training.recommender import compute_meta_features
from src.training.trainer import run_experiments
from src.tracking.tracker import track_experiment, load_champion

df = load_iris(as_frame=True).frame
target = "target"
types = profile_dataset(df)["column_types"]
info = detect_task(df, target)
meta = compute_meta_features(df, target, info["task"], types, info["imbalance_ratio"])
pre, _ = build_preprocessor(df, target, types)

results, best, le = run_experiments(df, target, info["task"], pre)
out = track_experiment(results, best, info["task"], "iris", meta)
print(out)

model = load_champion()
print("Champion prediction on 3 rows:", model.predict(df.drop(columns=[target]).head(3)))