"""
Download real F1 race + qualifying results (2022-2024) from GitHub
and build the processed parquet used by train.py.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --years 2022 2023 2024 --out data/processed/openf1.parquet
"""
import argparse
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/toUpperCase78/formula1-datasets/master"

RACE_FILE = {
    2019: "formula1_2019season_raceResults.csv",
    2020: "formula1_2020season_raceResults.csv",
    2021: "formula1_2021season_raceResults.csv",
    2022: "Formula1_2022season_raceResults.csv",
    2023: "Formula1_2023season_raceResults.csv",
    2024: "Formula1_2024season_raceResults.csv",
    2025: "Formula1_2025Season_RaceResults.csv",
}
QUALI_FILE = {
    2022: "Formula1_2022season_qualifyingResults.csv",
    2023: "Formula1_2023season_qualifyingResults.csv",
    2024: "Formula1_2024season_qualifyingResults.csv",
    2025: "Formula1_2025Season_QualifyingResults.csv",
}


def _fetch(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _lap_to_seconds(s) -> float:
    """Convert 'M:SS.mmm' or 'SS.mmm' lap-time string to float seconds."""
    if not isinstance(s, str) or not s.strip():
        return np.nan
    s = s.strip()
    try:
        if ":" in s:
            m, rest = s.split(":", 1)
            return int(m) * 60 + float(rest)
        return float(s)
    except Exception:
        return np.nan


def build_dataset(years: list[int], meeting_offset: int = 0) -> pd.DataFrame:
    all_rows = []
    offset = meeting_offset
    for year in sorted(years):
        if year not in RACE_FILE:
            print(f"  [skip] no race file configured for {year}")
            continue

        print(f"  Fetching {year}...")
        race = _fetch(f"{BASE_URL}/{RACE_FILE[year]}")

        tracks = race["Track"].unique()
        track_to_meeting = {t: offset + i + 1 for i, t in enumerate(tracks)}
        offset += len(tracks)

        race["year"] = year
        race["meeting_key"] = race["Track"].map(track_to_meeting)
        race["session_key_race"] = race["meeting_key"] * 100
        race["driver_number"] = pd.to_numeric(race["No"], errors="coerce")
        race["position"] = pd.to_numeric(race["Position"], errors="coerce")
        race["grid"] = pd.to_numeric(race.get("Starting Grid", pd.Series(dtype=float)), errors="coerce")
        race["dnf"] = race["Time/Retired"].apply(
            lambda x: 0 if re.match(r"^(\+|\d)", str(x)) else 1
        )
        race["team_name"] = race["Team"]
        race["circuit_short_name"] = race["Track"]
        race["number_of_laps"] = pd.to_numeric(race["Laps"], errors="coerce")
        race["gap_to_leader"] = race["Time/Retired"].astype(str)
        race["dns"] = 0
        race["dsq"] = 0
        race["duration"] = np.nan
        race["country_name"] = ""

        # Merge qualifying if available
        if year in QUALI_FILE:
            quali = _fetch(f"{BASE_URL}/{QUALI_FILE[year]}")
            for q in ["Q3", "Q2", "Q1"]:
                if q not in quali.columns:
                    quali[q] = np.nan
                quali[q] = quali[q].apply(_lap_to_seconds)
            quali["best_qual_sec"] = (
                quali["Q3"].combine_first(quali["Q2"]).combine_first(quali["Q1"])
            )
            quali["driver_number"] = pd.to_numeric(quali["No"], errors="coerce")
            quali["meeting_key"] = quali["Track"].map(track_to_meeting)
            race = race.merge(
                quali[["meeting_key", "driver_number", "best_qual_sec"]],
                on=["meeting_key", "driver_number"],
                how="left",
            )
        else:
            race["best_qual_sec"] = np.nan

        keep = [
            "year", "meeting_key", "session_key_race", "driver_number",
            "position", "grid", "best_qual_sec", "dnf", "dns", "dsq",
            "number_of_laps", "duration", "gap_to_leader",
            "team_name", "circuit_short_name", "country_name",
        ]
        all_rows.append(race[keep])

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df[df["position"].between(1, 20)].copy()
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024],
                    help="Seasons to download (default: 2022 2023 2024)")
    ap.add_argument("--out", default="data/processed/openf1.parquet",
                    help="Output parquet path")
    args = ap.parse_args()

    print(f"Downloading seasons: {args.years}")
    df = build_dataset(args.years)

    if df.empty:
        raise SystemExit("No data downloaded. Check the year range or network.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} rows across {df['meeting_key'].nunique()} races → {args.out}")


if __name__ == "__main__":
    main()
