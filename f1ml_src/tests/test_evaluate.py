import numpy as np
import pytest
from f1ml.evaluate import metrics


def test_metrics_mae_exact():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    m = metrics(y_true, y_pred)
    assert m["MAE"] == 0.0


def test_metrics_mae_nonzero():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 3, 4, 5])
    m = metrics(y_true, y_pred)
    assert m["MAE"] == pytest.approx(1.0)


def test_metrics_no_proba_no_topk():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    m = metrics(y_true, y_pred, proba=None)
    assert "Top-3" not in m
    assert "MAE" in m


def test_metrics_with_proba_topk():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    # Perfect proba: each row has 1.0 for the correct class
    proba = np.eye(3)
    m = metrics(y_true, y_pred, proba=proba, k=2)
    assert f"Top-2" in m
    assert m["Top-2"] == pytest.approx(1.0)


def test_metrics_returns_dict():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 3, 2])
    m = metrics(y_true, y_pred)
    assert isinstance(m, dict)
    assert "MAE" in m


def test_metrics_invalid_proba_does_not_crash():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    # proba with wrong shape should not raise (caught internally)
    bad_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    m = metrics(y_true, y_pred, proba=bad_proba, k=3)
    assert "MAE" in m
