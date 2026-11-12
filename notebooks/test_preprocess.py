import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from src.profiling.profiler import profile_dataset
from src.preprocessing.builder import build_preprocessor

n = 100
df = pd.DataFrame({
    "id": range(n),
    "age": np.where(np.arange(n) % 10 == 0, np.nan, np.random.randint(18, 80, n)),
    "city": np.where(np.arange(n) % 7 == 0, None, np.random.choice(["A", "B", "C"], n)),
    "signup": pd.date_range("2024-01-01", periods=n).astype(str),
    "bought": np.random.randint(0, 2, n),
})

types = profile_dataset(df)["column_types"]
pre, notes = build_preprocessor(df, "bought", types)
X = pre.fit_transform(df.drop(columns="bought"))

for note in notes:
    print(note)
print(X.shape)
print(X.head())
print("Missing values left:", int(X.isna().sum().sum()))