import sys
sys.path.append("..")
from sklearn.datasets import load_iris, fetch_california_housing
from src.profiling.profiler import profile_dataset
from src.profiling.task_detector import detect_task
from src.training.recommender import compute_meta_features, recommend


def show(df, target):
    types = profile_dataset(df)["column_types"]
    info = detect_task(df, target)
    meta = compute_meta_features(df, target, info["task"], types, info["imbalance_ratio"])
    print(meta)
    for i, r in enumerate(recommend(info["task"], meta), 1):
        print(f"{i}. {r['model']}  (score {r['score']})")
        for reason in r["reasons"]:
            print("     -", reason)
    print()


print("=== IRIS ===")
show(load_iris(as_frame=True).frame, "target")
print("=== HOUSING ===")
show(fetch_california_housing(as_frame=True).frame, "MedHouseVal")