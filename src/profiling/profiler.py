"""Dataset profiler: reads a DataFrame and summarises it."""
import warnings

import numpy as np
import pandas as pd


def detect_column_types(df: pd.DataFrame) -> dict:
    """Split columns into numerical, categorical, datetime and text."""
    types = {"numerical": [], "categorical": [], "datetime": [], "text": []}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            types["datetime"].append(col)
        elif pd.api.types.is_bool_dtype(series):
            types["categorical"].append(col)
        elif pd.api.types.is_numeric_dtype(series):
            types["numerical"].append(col)
        else:
            # Object/category columns: convert to string, then try parsing as dates
            as_str = series.dropna().astype(str)
            if len(as_str) == 0:
                types["categorical"].append(col)
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(as_str, errors="coerce")
            if parsed.notna().mean() > 0.9:
                types["datetime"].append(col)
            elif as_str.nunique() / len(as_str) > 0.5 and as_str.str.len().mean() > 30:
                types["text"].append(col)
            else:
                types["categorical"].append(col)
    return types


def missing_values(df: pd.DataFrame) -> dict:
    """Count and percentage of missing values per column."""
    counts = df.isna().sum()
    return {
        col: {"count": int(counts[col]), "percent": round(counts[col] / len(df) * 100, 2)}
        for col in df.columns
    }


def detect_outliers(df: pd.DataFrame, numeric_cols: list) -> dict:
    """Count outliers per numeric column using the IQR rule."""
    result = {}
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return result


def top_correlations(df: pd.DataFrame, numeric_cols: list, threshold: float = 0.8) -> list:
    """Pairs of numeric columns that are strongly correlated."""
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr().abs()
    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                pairs.append({"col_a": a, "col_b": b, "correlation": round(float(corr.loc[a, b]), 3)})
    return pairs


def profile_dataset(df: pd.DataFrame) -> dict:
    """Main entry point: returns a full profile of the dataset."""
    types = detect_column_types(df)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_types": types,
        "missing_values": missing_values(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
        "outliers": detect_outliers(df, types["numerical"]),
        "high_correlations": top_correlations(df, types["numerical"]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    }