import sys, json
sys.path.append("..")
import pandas as pd
from sklearn.datasets import load_iris
from src.profiling.profiler import profile_dataset

df = load_iris(as_frame=True).frame
print(json.dumps(profile_dataset(df), indent=2))