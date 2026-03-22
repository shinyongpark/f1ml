import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr


def metrics(y_true, y_pred, proba=None, k: int = 3) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    corr, _ = spearmanr(y_true, y_pred)
    spearman = float(corr) if not np.isnan(corr) else 0.0

    # Within-N: fraction of predictions within N positions of truth
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    within_3 = float(np.mean(np.abs(y_true - y_pred) <= 3))

    out = {
        "MAE": mae,
        "RMSE": rmse,
        "Spearman": spearman,
        "Within1": within_1,
        "Within3": within_3,
    }

    if proba is not None:
        try:
            from sklearn.metrics import top_k_accuracy_score
            out[f"Top-{k}"] = float(
                top_k_accuracy_score(y_true, proba, k=k, labels=np.unique(y_true))
            )
        except Exception:
            pass

    return out
