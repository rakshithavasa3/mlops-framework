"""Build an automatic preprocessing pipeline from a DataFrame."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DateFeatures(BaseEstimator, TransformerMixin):
    """Turn date columns into year, month, day and weekday numbers."""

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        out = {}
        for col in X.columns:
            d = pd.to_datetime(X[col], errors="coerce")
            out[f"{col}_year"] = d.dt.year
            out[f"{col}_month"] = d.dt.month
            out[f"{col}_day"] = d.dt.day
            out[f"{col}_weekday"] = d.dt.weekday
        return pd.DataFrame(out).astype(float)

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.columns_:
            names += [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday"]
        return np.array(names)


def _is_id_like(series: pd.Series) -> bool:
    n = len(series)
    return n > 20 and series.nunique() == n and (
        pd.api.types.is_integer_dtype(series) or series.dtype == object
    )


def build_preprocessor(df: pd.DataFrame, target: str, types: dict):
    """Return (ColumnTransformer, notes). `types` comes from profile_dataset()['column_types']."""
    notes = []
    features = [c for c in df.columns if c != target]

    numeric, categorical, dates = [], [], []
    for col in features:
        if _is_id_like(df[col]):
            notes.append(f"Dropped '{col}': every value is unique (looks like an ID).")
        elif col in types["text"]:
            notes.append(f"Dropped '{col}': free-text column.")
        elif col in types["datetime"]:
            dates.append(col)
        elif col in types["numerical"]:
            numeric.append(col)
        else:
            if df[col].nunique() > 50:
                notes.append(f"Dropped '{col}': {df[col].nunique()} categories is too many to encode.")
            else:
                categorical.append(col)

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if dates:
        transformers.append(("date", Pipeline([
            ("extract", DateFeatures()),
            ("impute", SimpleImputer(strategy="median")),
        ]), dates))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    notes.append(f"Numeric: {numeric}; Categorical: {categorical}; Dates: {dates}")
    return pre, notes