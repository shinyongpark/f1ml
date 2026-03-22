import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from f1ml.modeling import make_xy, build_model, _CANDIDATE_FEATURES


def _sample_df():
    return pd.DataFrame({
        "grid_fill": [1.0, 2.0, 3.0, 4.0],
        "best_qual_sec": [80.0, 80.5, 81.0, 81.5],
        "qual_rank_norm": [0.25, 0.5, 0.75, 1.0],
        "driver_form3": [2.0, 3.0, 4.0, 5.0],
        "team_strength": [-1.5, -2.0, -3.0, -3.5],
        "driver_season_points": [25, 18, 0, 0],
        "pos_num": [1, 2, 3, 4],
        "team_name": ["A", "B", "A", "B"],
    })


def test_make_xy_default_target():
    df = _sample_df()
    X, y = make_xy(df)
    assert list(y) == [1, 2, 3, 4]


def test_make_xy_uses_candidate_features():
    df = _sample_df()
    X, y = make_xy(df)
    # All candidate features that exist in df should be in X
    for col in ["grid_fill", "best_qual_sec", "qual_rank_norm", "driver_form3", "team_strength", "driver_season_points"]:
        assert col in X.columns, f"Expected feature '{col}' missing from X"


def test_make_xy_excludes_target():
    df = _sample_df()
    X, y = make_xy(df)
    assert "pos_num" not in X.columns


def test_make_xy_missing_target_raises():
    df = _sample_df().drop(columns=["pos_num"])
    with pytest.raises(KeyError):
        make_xy(df, target_col="pos_num")


def test_make_xy_includes_team_ohe():
    df = _sample_df()
    df["team_RedBull"] = [1, 0, 1, 0]
    X, y = make_xy(df)
    assert "team_RedBull" in X.columns


def test_make_xy_fallback_to_numeric():
    df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0], "pos_num": [1, 2]})
    X, y = make_xy(df)
    assert "feat_a" in X.columns
    assert "feat_b" in X.columns


def test_build_model_xgb():
    from xgboost import XGBRegressor
    model = build_model("xgb")
    assert isinstance(model, XGBRegressor)


def test_build_model_lgbm():
    from lightgbm import LGBMRegressor
    model = build_model("lgbm")
    assert isinstance(model, LGBMRegressor)


def test_build_model_rfr():
    model = build_model("rfr")
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 300


def test_build_model_gbr():
    model = build_model("gbr")
    assert isinstance(model, GradientBoostingRegressor)


def test_build_model_ridge():
    model = build_model("ridge")
    assert isinstance(model, Ridge)


def test_build_model_custom_params():
    model = build_model("rfr", n_estimators=50, random_state=7)
    assert model.n_estimators == 50
    assert model.random_state == 7


def test_build_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model kind"):
        build_model("svm")


def test_candidate_features_list_is_defined():
    assert isinstance(_CANDIDATE_FEATURES, list)
    assert len(_CANDIDATE_FEATURES) > 0
    assert "grid_fill" in _CANDIDATE_FEATURES
    assert "driver_form3" in _CANDIDATE_FEATURES
    assert "driver_season_points" in _CANDIDATE_FEATURES
