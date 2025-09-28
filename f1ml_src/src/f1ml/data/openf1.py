from __future__ import annotations
import time
from typing import Iterable, Dict, Any, List
from pathlib import Path
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.openf1.org/v1"

def _get(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GET with simple retries."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        try:
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []

def sessions(year: int, session_name: str | None = None, session_type: str | None = None) -> pd.DataFrame:
    params: Dict[str, Any] = {"year": year}
    if session_name:
        params["session_name"] = session_name
    if session_type:
        params["session_type"] = session_type
    data = _get("sessions", params)
    return pd.DataFrame(data)

def session_result(session_key: int) -> pd.DataFrame:
    data = _get("session_result", {"session_key": session_key})
    return pd.DataFrame(data)

def starting_grid(session_key: int) -> pd.DataFrame:
    data = _get("starting_grid", {"session_key": session_key})
    return pd.DataFrame(data)

def drivers(session_key: int) -> pd.DataFrame:
    data = _get("drivers", {"session_key": session_key})
    return pd.DataFrame(data)

def _pick_quali_session(sessions_df: pd.DataFrame, meeting_key: int) -> int | None:
    """
    For a given meeting, choose the qualifying session_key.
    Preference order:
      1) session_type == 'Qualifying'
      2) session_name contains 'Qualifying'
      3) latest session_key among candidates
    """
    if sessions_df.empty or "meeting_key" not in sessions_df.columns:
        return None
    cand = sessions_df[sessions_df["meeting_key"] == meeting_key].copy()
    if cand.empty:
        return None

    if "session_type" in cand.columns:
        mask = cand["session_type"].astype(str).str.fullmatch("Qualifying", case=False, na=False)
        if mask.any():
            cand = cand[mask]

    if len(cand) != 1:
        name_mask = cand.get("session_name", pd.Series([], dtype=object)).astype(str).str.contains("Qualifying", case=False, na=False)
        if name_mask.any():
            cand = cand[name_mask]

    if cand.empty:
        return None

    cand = cand.sort_values("session_key")
    return int(cand.iloc[-1]["session_key"])

def _best_quali_duration(val):
    """duration can be scalar or [Q1,Q2,Q3]; return the best (min) in seconds."""
    if isinstance(val, list):
        vals = [x for x in val if x is not None]
        return min(vals) if vals else np.nan
    return val

def build_dataset(years: Iterable[int]) -> pd.DataFrame:
    """
    Build one row per (race session_key, driver) with:
      - target: final race position
      - features: grid position (starting grid), best qualifying time, driver/team meta
    """
    rows = []
    for year in years:
        all_ses = sessions(year)  # Practice, Quali, Race, Sprint, etc.
        if all_ses.empty:
            continue

        race = all_ses[all_ses["session_name"].astype(str).str.fullmatch("Race", case=False, na=False)].copy()
        if race.empty:
            continue

        for rs in race.to_dict("records"):
            try:
                r_key = int(rs["session_key"])
                meeting_key = int(rs["meeting_key"])
                year_val = int(rs.get("year", year))
                circuit = rs.get("circuit_short_name")
                country = rs.get("country_name")

                # Race result (target)
                rr = session_result(r_key)
                if rr.empty:
                    continue
                rr["session_key_race"] = r_key
                rr["meeting_key"] = meeting_key
                rr["year"] = year_val
                rr["circuit_short_name"] = circuit
                rr["country_name"] = country

                # Starting grid
                sg = starting_grid(r_key)
                if not sg.empty:
                    sg = sg.rename(columns={"position": "grid"})
                    rr = rr.merge(
                        sg[["driver_number", "grid", "session_key"]].rename(columns={"session_key": "session_key_race"}),
                        on=["driver_number", "session_key_race"],
                        how="left",
                    )

                # Qualifying result for same meeting
                q_key = _pick_quali_session(all_ses, meeting_key)
                if q_key is not None:
                    qr = session_result(q_key)
                    if not qr.empty and "duration" in qr.columns:
                        qr["best_qual_sec"] = qr["duration"].apply(_best_quali_duration)
                        qr = qr[["driver_number", "best_qual_sec"]]
                        rr = rr.merge(qr, on="driver_number", how="left")

                # Driver/team info
                dr = drivers(r_key)
                if not dr.empty:
                    keepd = [c for c in ["driver_number","name_acronym","team_name","broadcast_name"] if c in dr.columns]
                    rr = rr.merge(dr[keepd], on="driver_number", how="left")

                # Normalize schema
                keep_cols = [
                    "year","meeting_key","session_key_race","driver_number","position","grid",
                    "dnf","dns","dsq","number_of_laps","duration","gap_to_leader",
                    "best_qual_sec","team_name","name_acronym","broadcast_name",
                    "circuit_short_name","country_name",
                ]
                for c in keep_cols:
                    if c not in rr.columns:
                        rr[c] = np.nan

                rows.append(rr[keep_cols])
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    # Basic types
    for c in ["position","grid","number_of_laps","driver_number","year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Sanitize problematic columns for Parquet ---
    # gap_to_leader can be '+1 LAP', '+12.345', None, etc. -> store as string to avoid Arrow coercion issues.
    if "gap_to_leader" in df.columns:
        df["gap_to_leader"] = df["gap_to_leader"].astype(str)

    # duration in race results may be numeric seconds or non-numeric -> coerce to float (seconds) when possible
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # best_qual_sec should be float
    if "best_qual_sec" in df.columns:
        df["best_qual_sec"] = pd.to_numeric(df["best_qual_sec"], errors="coerce")

    # Ensure any remaining object columns are strings (Arrow-friendly)
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        df[obj_cols] = df[obj_cols].astype(str)

    return df
