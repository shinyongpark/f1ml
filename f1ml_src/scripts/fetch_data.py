import argparse
from pathlib import Path
import pandas as pd
from f1ml.data.openf1 import build_dataset

def main(years, out_path: str):
    years = sorted(set(years))
    df = build_dataset(years)
    if df.empty:
        raise SystemExit("No data returned from OpenF1. Try a different year range.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True, help="e.g. --years 2023 2024 2025")
    ap.add_argument("--out", default="data/processed/openf1.parquet")
    args = ap.parse_args()
    main(args.years, args.out)
