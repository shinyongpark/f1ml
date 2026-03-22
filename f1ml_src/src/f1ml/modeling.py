import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Ordered list of candidate feature columns (engineered + raw).
# make_xy() takes all that exist in the dataframe.
_CANDIDATE_FEATURES = [
    # Qualifying & grid — strongest predictors
    "grid_fill",
    "grid",
    "grid_norm",
    "best_qual_sec",
    "best_qual_rank",
    "qual_rank_norm",
    # Team strength
    "team_strength",
    "team_mean_pos_year",
    # Driver recent form
    "driver_form3",
    "driver_form5",
    # Circuit-specific history
    "driver_circuit_avg",
    # Season context
    "driver_season_points",
    "driver_season_rank",
    # Reliability
    "driver_dnf_rate",
]


def make_xy(df: pd.DataFrame, target_col: str = "pos_num"):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")

    # Take all candidate features that are present
    cols = [c for c in _CANDIDATE_FEATURES if c in df.columns]

    # Also include team one-hot columns
    team_ohe = [c for c in df.columns if c.startswith("team_") and c not in cols]
    cols += team_ohe

    # Fallback: use all numeric columns except target
    if not cols:
        cols = (
            df.select_dtypes(include="number")
            .drop(columns=[target_col], errors="ignore")
            .columns.tolist()
        )

    X = df[cols].copy()
    y = df[target_col]
    print(f"Using features ({len(cols)}): {cols}")
    print(f"X shape: {X.shape}, y len: {len(y)}")
    return X, y


def build_model(kind: str = "xgb", **kwargs):
    if kind == "xgb":
        from xgboost import XGBRegressor
        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        defaults.update(kwargs)
        return XGBRegressor(**defaults)

    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(kwargs)
        return LGBMRegressor(**defaults)

    if kind == "rfr":
        defaults = dict(n_estimators=300, random_state=42, n_jobs=-1)
        defaults.update(kwargs)
        return RandomForestRegressor(**defaults)

    if kind == "gbr":
        defaults = dict(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
        defaults.update(kwargs)
        return GradientBoostingRegressor(**defaults)

    if kind == "ridge":
        defaults = dict(alpha=1.0)
        defaults.update(kwargs)
        return Ridge(**defaults)

    raise ValueError(f"Unknown model kind: {kind!r}. Choose from: xgb, lgbm, rfr, gbr, ridge")
