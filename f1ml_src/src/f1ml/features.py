import pandas as pd
import numpy as np

def make_weekend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # existing
    if "grid" in df.columns:
        df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
        df["grid_fill"] = df["grid"].fillna(df["grid"].median())
    if "best_qual_sec" in df.columns:
        df["best_qual_sec"] = pd.to_numeric(df["best_qual_sec"], errors="coerce")
        # normalize quali pace *within the race* to remove unit/track effects
        df["best_qual_rank"] = df.groupby("session_key_race")["best_qual_sec"].rank(method="min")

    # team strength proxy (within same year)
    if {"team_name","pos_num","year"}.issubset(df.columns):
        df["team_mean_pos_year"] = (
            df.groupby(["year","team_name"])["pos_num"].transform("mean")
        )
        # lower is better → invert so higher is stronger
        df["team_strength"] = -df["team_mean_pos_year"]

    # driver form: rolling average finish over last N races (per driver)
    if {"driver_number","pos_num","year","meeting_key"}.issubset(df.columns):
        df = df.sort_values(["driver_number","year","meeting_key"])
        df["driver_form3"] = (
            df.groupby("driver_number")["pos_num"]
              .apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )
        df["driver_form5"] = (
            df.groupby("driver_number")["pos_num"]
              .apply(lambda s: s.shift().rolling(5, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )

    # grid-based deltas
    if {"grid","pos_num"}.issubset(df.columns):
        df["delta_grid_pos"] = pd.to_numeric(df["pos_num"], errors="coerce") - df["grid_fill"]

    # one-hot team (small cardinality)
    if "team_name" in df.columns:
        ohe = pd.get_dummies(df["team_name"], prefix="team", dummy_na=False)
        df = pd.concat([df, ohe], axis=1)

    # Team strength (lower mean pos is better → invert)
    if {"team_name","pos_num","year"}.issubset(df.columns):
        df["team_mean_pos_year"] = (
            df.groupby(["year","team_name"])["pos_num"].transform("mean")
        )
        df["team_strength"] = -df["team_mean_pos_year"]

    # Driver form (last 3 / 5 finishes)
    if {"driver_number","pos_num","year","meeting_key"}.issubset(df.columns):
        df = df.sort_values(["driver_number","year","meeting_key"])
        df["driver_form3"] = (
            df.groupby("driver_number")["pos_num"]
            .apply(lambda s: s.shift().rolling(3, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df["driver_form5"] = (
            df.groupby("driver_number")["pos_num"]
            .apply(lambda s: s.shift().rolling(5, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )


    # final simple null handling
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df

def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    # leave for circuit similarity later
    return df
