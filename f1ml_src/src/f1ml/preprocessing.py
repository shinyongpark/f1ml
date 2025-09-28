from pathlib import Path
import pandas as pd

def load_raw(results_csv: str, races_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(results_csv)
    races = pd.read_csv(races_csv)
    return results, races

def basic_clean(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    df = results.merge(races, on="raceId", how="left") if "raceId" in results.columns and "raceId" in races.columns else results.copy()
    if "positionOrder" in df.columns:
        df = df.dropna(subset=["positionOrder"])
    return df

def split_by_year(df: pd.DataFrame, year_col: str = "year"):
    if year_col not in df.columns:
        return df, df.iloc[0:0]
    train = df[df[year_col] < 2025]
    test  = df[df[year_col] == 2025]
    return train, test

def save_parquet(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
