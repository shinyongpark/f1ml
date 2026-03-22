import pandas as pd
import numpy as np
import pytest
from f1ml.features import make_weekend_features, add_history_features


def _base_df():
    """Minimal dataframe with all required columns."""
    return pd.DataFrame({
        "driver_number": [1, 2, 3, 1, 2, 3],
        "year": [2023, 2023, 2023, 2024, 2024, 2024],
        "meeting_key": [1, 1, 1, 2, 2, 2],
        "session_key_race": [10, 10, 10, 20, 20, 20],
        "pos_num": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0],
        "grid": [2.0, 1.0, 3.0, 1.0, 3.0, 2.0],
        "best_qual_sec": [80.1, 80.5, 81.0, 79.8, 80.2, 80.9],
        "team_name": ["RedBull", "Mercedes", "Ferrari", "RedBull", "Mercedes", "Ferrari"],
        "circuit_short_name": ["Bahrain"] * 3 + ["Monaco"] * 3,
        "dnf": [0, 0, 1, 0, 0, 0],
    })


# --- make_weekend_features ---

def test_returns_copy():
    df = _base_df()
    result = make_weekend_features(df)
    assert result is not df


def test_grid_fill_imputes_nans():
    df = _base_df()
    df.loc[0, "grid"] = np.nan
    result = make_weekend_features(df)
    assert "grid_fill" in result.columns
    assert result["grid_fill"].notna().all()


def test_qual_rank_computed():
    df = _base_df()
    result = make_weekend_features(df)
    assert "best_qual_rank" in result.columns
    assert "qual_rank_norm" in result.columns
    # qual_rank_norm should be in (0, 1]
    assert (result["qual_rank_norm"] > 0).all()
    assert (result["qual_rank_norm"] <= 1).all()


def test_grid_norm_computed():
    df = _base_df()
    result = make_weekend_features(df)
    assert "grid_norm" in result.columns
    assert (result["grid_norm"] > 0).all()


def test_team_strength_is_negative_mean_pos():
    df = _base_df()
    result = make_weekend_features(df)
    assert "team_strength" in result.columns
    pd.testing.assert_series_equal(
        result["team_strength"], -result["team_mean_pos_year"], check_names=False
    )


def test_driver_form_columns_exist():
    df = _base_df()
    result = make_weekend_features(df)
    assert "driver_form3" in result.columns
    assert "driver_form5" in result.columns


def test_team_ohe_columns():
    df = _base_df()
    result = make_weekend_features(df)
    assert "team_RedBull" in result.columns
    assert "team_Mercedes" in result.columns
    assert "team_Ferrari" in result.columns


def test_no_nulls_in_numeric_after_imputation():
    df = _base_df()
    df.loc[1, "best_qual_sec"] = np.nan
    result = make_weekend_features(df)
    num_cols = result.select_dtypes(include="number").columns
    assert result[num_cols].isna().sum().sum() == 0


def test_no_duplicate_columns():
    df = _base_df()
    result = make_weekend_features(df)
    assert result.columns.duplicated().sum() == 0


def test_no_delta_grid_pos_data_leakage():
    """delta_grid_pos used the target column — it must not be present."""
    df = _base_df()
    result = make_weekend_features(df)
    assert "delta_grid_pos" not in result.columns


# --- add_history_features ---

def test_add_history_driver_circuit_avg():
    df = _base_df()
    df = make_weekend_features(df)
    result = add_history_features(df)
    assert "driver_circuit_avg" in result.columns


def test_add_history_dnf_rate():
    df = _base_df()
    df = make_weekend_features(df)
    result = add_history_features(df)
    assert "driver_dnf_rate" in result.columns
    assert (result["driver_dnf_rate"] >= 0).all()
    assert (result["driver_dnf_rate"] <= 1).all()


def test_add_history_season_points():
    df = _base_df()
    df = make_weekend_features(df)
    result = add_history_features(df)
    assert "driver_season_points" in result.columns
    # Points should be non-negative
    assert (result["driver_season_points"] >= 0).all()


def test_add_history_season_rank():
    df = _base_df()
    df = make_weekend_features(df)
    result = add_history_features(df)
    assert "driver_season_rank" in result.columns


def test_history_no_future_leakage():
    """First race of the season should have 0 season points (no prior races)."""
    df = _base_df()
    df = make_weekend_features(df)
    result = add_history_features(df)
    # Drivers in meeting_key=1 (first race) should have 0 season points
    first_race = result[result["meeting_key"] == 1]
    assert (first_race["driver_season_points"] == 0).all()


def test_add_history_no_nulls_in_numeric():
    df = _base_df()
    df = make_weekend_features(df)
    result = add_history_features(df)
    num_cols = result.select_dtypes(include="number").columns
    assert result[num_cols].isna().sum().sum() == 0
