import argparse
from pathlib import Path
import pandas as pd
from joblib import dump
import yaml

from f1ml.preprocessing import (
    load_raw,
    basic_clean,
    split_by_year,
)
from f1ml.features import make_weekend_features, add_history_features
from f1ml.modeling import make_xy, build_model
from f1ml.evaluate import metrics



def _clean_xy(X, y):
    import numpy as np
    import pandas as pd

    # drop rows with NaN target
    mask = y.notna() & ~pd.isna(y)
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # cast y to int if numeric-like
    y = pd.to_numeric(y, errors="coerce")
    mask2 = y.notna()
    X = X.loc[mask2].copy()
    y = y.loc[mask2].astype(int)

    # impute numeric NaNs in X with column medians
    num_cols = X.select_dtypes(include="number").columns
    if len(num_cols):
        meds = X[num_cols].median()
        X[num_cols] = X[num_cols].fillna(meds)

    # drop columns that are entirely NaN
    X = X.loc[:, X.notna().any(axis=0)].copy()
    # If any non-numeric columns slipped in, try to coerce
    for c in X.columns:
        if X[c].dtype.kind not in 'biufc':
            X[c] = pd.to_numeric(X[c], errors='coerce')
    # impute again in case coercion created NaNs
    num_cols = X.select_dtypes(include='number').columns
    if len(num_cols):
        meds = X[num_cols].median()
        X[num_cols] = X[num_cols].fillna(meds)
    y = y.loc[X.index]
    return X, y
def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg   = cfg.get("data", {})
    schema_cfg = cfg.get("schema", {})
    model_cfg  = cfg.get("model", {})
    eval_cfg   = cfg.get("eval", {})
    out_dir    = Path(cfg.get("output_dir", "artifacts"))

    processed_parquet = data_cfg.get("processed_parquet")

    # --- Load data (prefer processed_parquet if provided) ---
    if processed_parquet:
        try:
            df = pd.read_parquet(processed_parquet)
        except FileNotFoundError:
            raise SystemExit(
                f"Processed parquet not found at {processed_parquet}.\n"
                "Run the fetch step first, e.g.:\n"
                "  python scripts/fetch_data.py --years 2023 2024 2025 --out data/processed/openf1.parquet"
            )
    else:
        results_csv = data_cfg.get("results_csv")
        races_csv   = data_cfg.get("races_csv")
        if not results_csv or not races_csv:
            raise SystemExit("CSV paths not provided in config and no processed_parquet specified.")
        df = basic_clean(*load_raw(results_csv, races_csv))
        if df.empty:
            raise SystemExit("Loaded dataframe is empty. Check your data config or fetch step.")

    # --- Feature engineering ---
    df = make_weekend_features(df)
    df = add_history_features(df)

    # --- Target coercion: use numeric finishing position 1..20 ---
    target_col_cfg = schema_cfg.get('target_col', 'position')
    if target_col_cfg not in df.columns:
        raise SystemExit(f"Target column '{target_col_cfg}' not found in dataframe.")
    df['pos_num'] = pd.to_numeric(df[target_col_cfg], errors='coerce')
    df = df[(df['pos_num'] >= 1) & (df['pos_num'] <= 20)].copy()

    # --- Prefilter: require target + at least one useful numeric feature ---
    target_col = 'pos_num'
    # If we have grid/best_qual_sec, require at least one of them; otherwise, ensure at least one numeric feature exists
    preferred = [c for c in ['grid','best_qual_sec'] if c in df.columns]
    if preferred:
        df = df[df[preferred].notna().any(axis=1)].copy()
    else:
        num_cols_all = df.select_dtypes(include='number').columns.tolist()
        num_cols_all = [c for c in num_cols_all if c != target_col]
        if not num_cols_all:
            raise SystemExit('No numeric features available to train on.')

        print(f"Rows after target+feature filters: {len(df)}")
    print("Numeric columns (sample):", df.select_dtypes(include='number').columns.tolist()[:12])
    print("Has features:", [c for c in ['grid','best_qual_sec','qual_rank_norm'] if c in df.columns])
    
    # --- Split ---
    year_col   = schema_cfg.get('year_col', 'year')
    target_col = 'pos_num'
    train, test = split_by_year(df, year_col=year_col)

    if train.empty or test.empty:
        # Fallback: simple 80/20 split
        n = len(df)
        split_idx = max(1, int(n * 0.8))
        train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    # If still empty after fallback, bail early with a clear message
    if train.empty or test.empty:
        raise SystemExit('Not enough samples after filtering to perform a train/test split.')

    # --- Model ---
    Xtr, ytr = make_xy(train, target_col=target_col)
    Xtr, ytr = _clean_xy(Xtr, ytr)

    Xte, yte = make_xy(test,  target_col=target_col)
    Xte, yte = _clean_xy(Xte, yte)

    # Baseline: predict finishing position = starting grid (or 20 if NaN)
    if "grid" in test.columns:
        baseline = test["grid"].fillna(20).clip(1, 20).astype(int)
        from sklearn.metrics import mean_absolute_error
        print("Baseline(grid) MAE:", mean_absolute_error(yte, baseline))


    model = build_model(model_cfg.get("kind", "rf"), **model_cfg.get("params", {}))
    model.fit(Xtr, ytr)

    yhat = model.predict(Xte)
    proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None

    m = metrics(yte, yhat, proba=proba, k=eval_cfg.get("topk", 3))

    # --- Save artifacts ---
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(model, out_dir / "model.joblib")
    pd.DataFrame({"metric": list(m.keys()), "value": list(m.values())}).to_csv(out_dir / "metrics.csv", index=False)
    print("Metrics:", m)

    # in scripts/train.py, after features and filtering
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error
    groups = df["meeting_key"]
    gkf = GroupKFold(n_splits=5)

    # choose feature set once
    X_all, y_all = make_xy(df, target_col="pos_num")

    fold_mae = []
    for tr_idx, te_idx in gkf.split(X_all, y_all, groups):
        Xtr, ytr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
        Xte, yte = X_all.iloc[te_idx], y_all.iloc[te_idx]
        # quick clean (reuse helper)
        Xtr, ytr = _clean_xy(Xtr, ytr)
        Xte, yte = _clean_xy(Xte, yte)
        model = build_model("rf", n_estimators=600, max_depth=None, random_state=42)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        fold_mae.append(mean_absolute_error(yte, yhat))
    print("GroupKFold MAE (by race):", sum(fold_mae)/len(fold_mae))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/openf1_train.yaml")
    args = ap.parse_args()
    main(args.config)
