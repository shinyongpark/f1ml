import pandas as pd
import numpy as np
import pytest
from f1ml.features import make_weekend_features, add_history_features


def _base_df():
    """Minimal dataframe with all required columns for feature engineering."""
    return pd.DataFrame({
        "driver_number": [1, 2, 3, 1, 2, 3],
        "year": [2023, 2023, 2023, 2024, 2024, 2024],
        "meeting_key": [1, 1, 1, 2, 2, 2],
        "session_key_race": [10, 10, 10, 20, 20, 20],
        "pos_num": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0],
        "grid": [2.0, 1.0, 3.0, 1.0, 3.0, 2.0],
        "best_qual_sec": [80.1, 80.5, 81.0, 79.8, 80.2, 80.9],
        "team_name": ["RedBull", "Mercedes", "Ferrari", "RedBull", "Mercedes", "Ferrari"],
    })


def test_make_weekend_features_returns_copy():
    df = _base_df()
    result = make_weekend_features(df)
    assert result is not df


def test_make_weekend_features_grid_fill():
    df = _base_df()
    df.loc[0, "grid"] = np.nan
    result = make_weekend_features(df)
    assert "grid_fill" in result.columns
    assert result["grid_fill"].notna().all()


def test_make_weekend_features_best_qual_rank():
    df = _base_df()
    result = make_weekend_features(df)
    assert "best_qual_rank" in result.columns
    # Within each race, ranks should be 1..n
    for key, grp in result.groupby("session_key_race"):
        assert set(grp["best_qual_rank"]) == {1.0, 2.0, 3.0}


def test_make_weekend_features_team_strength():
    df = _base_df()
    result = make_weekend_features(df)
    assert "team_mean_pos_year" in result.columns
    assert "team_strength" in result.columns
    # team_strength is negative of team_mean_pos_year
    pd.testing.assert_series_equal(
        result["team_strength"],
        -result["team_mean_pos_year"],
        check_names=False,
    )


def test_make_weekend_features_driver_form():
    df = _base_df()
    result = make_weekend_features(df)
    assert "driver_form3" in result.columns
    assert "driver_form5" in result.columns


def test_make_weekend_features_team_ohe():
    df = _base_df()
    result = make_weekend_features(df)
    assert "team_RedBull" in result.columns
    assert "team_Mercedes" in result.columns
    assert "team_Ferrari" in result.columns


def test_make_weekend_features_no_nulls_in_numeric():
    df = _base_df()
    df.loc[1, "best_qual_sec"] = np.nan
    result = make_weekend_features(df)
    num_cols = result.select_dtypes(include="number").columns
    assert result[num_cols].isna().sum().sum() == 0


def test_make_weekend_features_no_duplicate_columns():
    df = _base_df()
    result = make_weekend_features(df)
    assert result.columns.duplicated().sum() == 0, "Duplicate columns found"


def test_add_history_features_passthrough():
    df = _base_df()
    result = add_history_features(df)
    pd.testing.assert_frame_equal(result, df)
