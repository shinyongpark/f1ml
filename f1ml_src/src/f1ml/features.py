import pandas as pd
import numpy as np


_F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def _points_from_pos(pos) -> int:
    try:
        return _F1_POINTS.get(int(pos), 0)
    except (TypeError, ValueError):
        return 0


def make_weekend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Grid position ---
    if "grid" in df.columns:
        df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
        df["grid_fill"] = df["grid"].fillna(df["grid"].median())
        # Normalized grid rank within each race (0=pole, 1=last)
        n_drivers = df.groupby("session_key_race")["driver_number"].transform("count") if "session_key_race" in df.columns else len(df)
        df["grid_norm"] = df["grid_fill"] / n_drivers

    # --- Qualifying pace ---
    if "best_qual_sec" in df.columns:
        df["best_qual_sec"] = pd.to_numeric(df["best_qual_sec"], errors="coerce")
        # Rank within race (1=fastest)
        if "session_key_race" in df.columns:
            df["best_qual_rank"] = df.groupby("session_key_race")["best_qual_sec"].rank(method="min")
            n_drivers = df.groupby("session_key_race")["driver_number"].transform("count")
            df["qual_rank_norm"] = df["best_qual_rank"] / n_drivers

    # --- Team strength: mean finishing position for this team this year ---
    if {"team_name", "pos_num", "year"}.issubset(df.columns):
        df["team_mean_pos_year"] = (
            df.groupby(["year", "team_name"])["pos_num"].transform("mean")
        )
        # lower pos is better → invert so higher = stronger
        df["team_strength"] = -df["team_mean_pos_year"]

    # --- Driver recent form: rolling avg finish over last N races ---
    if {"driver_number", "pos_num", "year", "meeting_key"}.issubset(df.columns):
        df = df.sort_values(["driver_number", "year", "meeting_key"])
        df["driver_form3"] = (
            df.groupby("driver_number")["pos_num"]
            .transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
        )
        df["driver_form5"] = (
            df.groupby("driver_number")["pos_num"]
            .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        )

    # --- One-hot encode team (low cardinality) ---
    if "team_name" in df.columns:
        ohe = pd.get_dummies(df["team_name"], prefix="team", dummy_na=False)
        df = pd.concat([df, ohe], axis=1)

    # --- Null imputation for all numeric columns ---
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features derived from multi-race history:
      - driver_circuit_avg: driver's historical avg finish at this circuit
      - driver_dnf_rate: driver's historical DNF rate
      - driver_season_points: cumulative F1 points before this race
      - driver_season_rank: championship rank before this race
    All use shift() so only past races are used — no leakage.
    """
    df = df.copy()

    required_sort = {"driver_number", "year", "meeting_key"}

    if required_sort.issubset(df.columns):
        df = df.sort_values(["year", "meeting_key", "driver_number"]).reset_index(drop=True)

    # --- Driver historical avg finish at this circuit ---
    if {"driver_number", "circuit_short_name", "pos_num"}.issubset(df.columns):
        df["driver_circuit_avg"] = (
            df.groupby(["driver_number", "circuit_short_name"])["pos_num"]
            .transform(lambda s: s.shift().expanding(min_periods=1).mean())
        )

    # --- Driver DNF rate (historical) ---
    if {"driver_number", "dnf"}.issubset(df.columns):
        df["dnf"] = pd.to_numeric(df["dnf"], errors="coerce").fillna(0)
        df["driver_dnf_rate"] = (
            df.groupby("driver_number")["dnf"]
            .transform(lambda s: s.shift().expanding(min_periods=1).mean())
        )

    # --- Cumulative season points before this race ---
    if {"driver_number", "pos_num", "year", "meeting_key"}.issubset(df.columns):
        df["_race_pts"] = df["pos_num"].apply(_points_from_pos)
        df["driver_season_points"] = (
            df.groupby(["year", "driver_number"])["_race_pts"]
            .transform(lambda s: s.shift(fill_value=0).cumsum())
        )
        df = df.drop(columns=["_race_pts"])

        # Championship rank at time of each race (higher points = rank 1)
        df["driver_season_rank"] = (
            df.groupby(["year", "meeting_key"])["driver_season_points"]
            .rank(method="min", ascending=False)
        )

    # --- Null imputation for any new numeric columns ---
    # For driver_circuit_avg, fall back to recent form before global median
    if "driver_circuit_avg" in df.columns:
        fallback = df.get("driver_form5", df.get("driver_form3", None))
        if fallback is not None:
            df["driver_circuit_avg"] = df["driver_circuit_avg"].fillna(fallback)

    num_cols = df.select_dtypes(include="number").columns
    # fillna(10.0) as last resort for all-NaN columns (e.g., first circuit appearances)
    medians = df[num_cols].median().fillna(10.0)
    df[num_cols] = df[num_cols].fillna(medians)

    return df
