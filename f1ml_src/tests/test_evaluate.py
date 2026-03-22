import numpy as np
import pytest
from f1ml.evaluate import metrics


def test_mae_perfect():
    y = np.array([1, 2, 3, 4])
    m = metrics(y, y)
    assert m["MAE"] == pytest.approx(0.0)


def test_mae_nonzero():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 3, 4, 5])
    m = metrics(y_true, y_pred)
    assert m["MAE"] == pytest.approx(1.0)


def test_rmse_perfect():
    y = np.array([1, 2, 3, 4])
    m = metrics(y, y)
    assert m["RMSE"] == pytest.approx(0.0)


def test_rmse_nonzero():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 3, 4, 5])
    m = metrics(y_true, y_pred)
    assert m["RMSE"] == pytest.approx(1.0)


def test_spearman_perfect():
    y = np.array([1, 2, 3, 4, 5])
    m = metrics(y, y)
    assert m["Spearman"] == pytest.approx(1.0)


def test_spearman_reversed():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])
    m = metrics(y_true, y_pred)
    assert m["Spearman"] == pytest.approx(-1.0)


def test_within1():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 5, 4])  # |5-3|=2, so 3/4 are within 1
    m = metrics(y_true, y_pred)
    assert m["Within1"] == pytest.approx(3 / 4)


def test_within3_all_match():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    m = metrics(y_true, y_pred)
    assert m["Within3"] == pytest.approx(1.0)


def test_no_proba_no_topk():
    y = np.array([1, 2, 3])
    m = metrics(y, y, proba=None)
    assert "Top-3" not in m


def test_all_keys_present():
    y = np.array([1, 2, 3])
    m = metrics(y, y)
    for key in ["MAE", "RMSE", "Spearman", "Within1", "Within3"]:
        assert key in m


def test_returns_dict():
    y = np.array([1, 2, 3])
    m = metrics(y, y)
    assert isinstance(m, dict)


def test_bad_proba_does_not_crash():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    bad_proba = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    m = metrics(y_true, y_pred, proba=bad_proba, k=3)
    assert "MAE" in m
