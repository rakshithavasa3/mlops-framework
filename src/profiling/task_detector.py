"""Decide classification vs regression and warn about bad targets."""
import pandas as pd


def detect_task(df: pd.DataFrame, target: str) -> dict:
    """Return the task type plus any warnings/errors for the chosen target."""
    if target not in df.columns:
        return {"task": None, "valid": False, "errors": [f"Column '{target}' not found."], "warnings": []}

    y = df[target]
    n = len(y)
    n_unique = y.nunique(dropna=True)
    errors, warnings = [], []
    task = None

    # 1. Hard blockers
    if pd.api.types.is_datetime64_any_dtype(y):
        errors.append("Target is a date/time column. Dates are not a valid prediction target; "
                      "extract features such as year or month, or choose another column.")
    elif y.dtype == object:
        parsed = pd.to_datetime(y.dropna().astype(str), errors="coerce")
        if len(parsed) > 0 and parsed.notna().mean() > 0.9:
            errors.append("Target looks like a date/time column. Choose a different target.")

    if n_unique <= 1:
        errors.append("Target has only one unique value, so there is nothing to predict.")

    if y.isna().mean() > 0.5:
        errors.append("More than 50% of the target values are missing.")

    # 2. Possible ID column
    if n_unique == n and (pd.api.types.is_integer_dtype(y) or y.dtype == object):
        errors.append("Every target value is unique, so this looks like an ID column.")

    # 3. Task type
    if not errors:
        is_numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
        if not is_numeric:
            task = "classification"
            if n_unique > 50:
                errors.append(f"Target has {n_unique} distinct classes, which is too many for classification.")
        else:
            is_int_like = pd.api.types.is_integer_dtype(y) or (y.dropna() % 1 == 0).all()
            if is_int_like and n_unique <= 20 and n_unique / n < 0.05:
                task = "classification"
            else:
                task = "regression"

    # 4. Class imbalance warning
    imbalance = None
    if task == "classification" and not errors:
        counts = y.value_counts(normalize=True)
        imbalance = round(float(counts.max() / counts.min()), 2)
        if imbalance > 3:
            warnings.append(f"Class imbalance detected (largest class is {imbalance}x the smallest). "
                            "Use F1/ROC-AUC and consider class weights.")
        if counts.min() * n < 10:
            warnings.append("Some classes have fewer than 10 samples; cross-validation may be unreliable.")

    return {
        "task": task,
        "valid": len(errors) == 0,
        "target": target,
        "unique_values": int(n_unique),
        "imbalance_ratio": imbalance,
        "errors": errors,
        "warnings": warnings,
    }