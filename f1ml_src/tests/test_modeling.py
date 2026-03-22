import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from f1ml.modeling import make_xy, build_model


def _sample_df():
    return pd.DataFrame({
        "grid": [1.0, 2.0, 3.0, 4.0],
        "best_qual_sec": [80.0, 80.5, 81.0, 81.5],
        "pos_num": [1, 2, 3, 4],
        "team_name": ["A", "B", "A", "B"],
    })


def test_make_xy_default_target():
    df = _sample_df()
    X, y = make_xy(df)
    assert list(y) == [1, 2, 3, 4]


def test_make_xy_uses_engineered_cols_first():
    df = _sample_df()
    X, y = make_xy(df)
    # Should prefer grid and best_qual_sec over raw numeric columns
    assert "grid" in X.columns
    assert "best_qual_sec" in X.columns
    # target should NOT be in X
    assert "pos_num" not in X.columns


def test_make_xy_missing_target_raises():
    df = _sample_df().drop(columns=["pos_num"])
    with pytest.raises(KeyError):
        make_xy(df, target_col="pos_num")


def test_make_xy_custom_target():
    df = _sample_df()
    df["custom_target"] = [10, 20, 30, 40]
    X, y = make_xy(df, target_col="custom_target")
    assert list(y) == [10, 20, 30, 40]


def test_build_model_rf_defaults():
    model = build_model("rf")
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 300
    assert model.random_state == 42


def test_build_model_rf_custom_params():
    model = build_model("rf", n_estimators=50, random_state=0)
    assert model.n_estimators == 50
    assert model.random_state == 0


def test_build_model_logreg_defaults():
    model = build_model("logreg")
    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 1000


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model kind"):
        build_model("xgboost")


def test_make_xy_fallback_to_all_numeric():
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0],
        "feature_b": [4.0, 5.0, 6.0],
        "pos_num": [1, 2, 3],
    })
    X, y = make_xy(df)
    assert "feature_a" in X.columns
    assert "feature_b" in X.columns
