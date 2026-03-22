import pandas as pd
import pytest
from f1ml.preprocessing import basic_clean, split_by_year


def _make_results():
    return pd.DataFrame({
        "raceId": [1, 2, 3],
        "driverId": [10, 20, 30],
        "positionOrder": [1.0, 2.0, None],
    })


def _make_races():
    return pd.DataFrame({
        "raceId": [1, 2, 3],
        "year": [2023, 2024, 2025],
        "name": ["Bahrain", "Australia", "Monaco"],
    })


def test_basic_clean_merges_on_raceId():
    results = _make_results()
    races = _make_races()
    df = basic_clean(results, races)
    assert "year" in df.columns
    assert "name" in df.columns


def test_basic_clean_drops_null_position():
    results = _make_results()
    races = _make_races()
    df = basic_clean(results, races)
    # Row with positionOrder=None should be dropped
    assert df["positionOrder"].notna().all()
    assert len(df) == 2


def test_basic_clean_no_raceId_returns_results():
    results = pd.DataFrame({"driverId": [1, 2], "position": [1, 2]})
    races = pd.DataFrame({"year": [2023, 2024]})
    df = basic_clean(results, races)
    assert list(df.columns) == list(results.columns)


def test_split_by_year_default():
    df = pd.DataFrame({"year": [2022, 2023, 2024, 2025, 2025], "x": range(5)})
    train, test = split_by_year(df)
    assert (train["year"] < 2025).all()
    assert (test["year"] == 2025).all()
    assert len(train) == 3
    assert len(test) == 2


def test_split_by_year_missing_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    train, test = split_by_year(df)
    assert len(train) == 3
    assert len(test) == 0


def test_split_by_year_custom_column():
    df = pd.DataFrame({"season": [2023, 2024, 2025], "x": [1, 2, 3]})
    train, test = split_by_year(df, year_col="season")
    assert len(train) == 2
    assert len(test) == 1
