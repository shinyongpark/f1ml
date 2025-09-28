import argparse
from joblib import load
import pandas as pd
from f1ml.modeling import make_xy
from f1ml.evaluate import metrics

def main(model_path: str, test_parquet: str, target_col: str = "positionOrder"):
    model = load(model_path)
    df = pd.read_parquet(test_parquet)
    X, y = make_xy(df, target_col=target_col)
    yhat = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    print(metrics(y, yhat, proba=proba))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--test",  default="data/processed/test.parquet")
    args = ap.parse_args()
    main(args.model, args.test)
