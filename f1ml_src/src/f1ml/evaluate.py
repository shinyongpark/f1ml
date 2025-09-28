import numpy as np
from sklearn.metrics import mean_absolute_error, top_k_accuracy_score

def metrics(y_true, y_pred, proba=None, k=3) -> dict:
    out = {"MAE": float(mean_absolute_error(y_true, y_pred))}
    if proba is not None:
        try:
            out[f"Top-{k}"] = float(top_k_accuracy_score(y_true, proba, k=k, labels=np.unique(y_true)))
        except Exception:
            pass
    return out
