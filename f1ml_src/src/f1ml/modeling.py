import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def make_xy(df: pd.DataFrame, target_col: str = "pos_num"):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")
    # Feature selection: if we already have engineered cols, prefer them
    cols = [c for c in ["qual_rank_norm","best_qual_sec","grid"] if c in df.columns]
    if not cols:
        cols = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore").columns.tolist()
    X = df[cols].copy()
    y = df[target_col]
    print(f"Using features ({len(cols)}):", cols)
    print(f"X shape: {X.shape}, y len: {len(y)}")
    return X, y

def build_model(kind: str = "rf", **kwargs):
    if kind == "logreg":
        # default max_iter unless overridden
        if "max_iter" not in kwargs:
            kwargs["max_iter"] = 1000
        return LogisticRegression(**kwargs)
    if kind == "rf":
        # default n_estimators/random_state unless overridden
        if "n_estimators" not in kwargs:
            kwargs["n_estimators"] = 300
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        return RandomForestClassifier(**kwargs)
    raise ValueError(f"Unknown model kind: {kind}")
