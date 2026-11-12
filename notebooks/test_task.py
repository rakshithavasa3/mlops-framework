import sys
sys.path.append("..")
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from src.profiling.task_detector import detect_task

iris = load_iris(as_frame=True).frame
print("Iris target:", detect_task(iris, "target"))

house = fetch_california_housing(as_frame=True).frame
print("Housing target:", detect_task(house, "MedHouseVal"))

# The bug from the reference screenshots
returns = pd.DataFrame({
    "TerritoryKey": [1, 2, 3, 4] * 50,
    "ReturnDate": pd.date_range("2020-01-01", periods=200).astype(str),
})
print("Date target:", detect_task(returns, "ReturnDate"))