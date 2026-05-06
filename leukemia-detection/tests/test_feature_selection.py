"""
test_feature_selection.py
=========================
Unit tests for src/feature_selection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.feature_selection import (
    AnovaSelector,
    TopVariableSelector,
    early_fusion,
    select_features,
)


@pytest.fixture
def XY():
    rng = np.random.default_rng(42)
    X = rng.random((100, 200))
    y = (rng.random(100) > 0.5).astype(int)
    return X, y


def test_anova_selector_output_shape(XY):
    X, y = XY
    sel = AnovaSelector(k=50)
    X_sel = sel.fit_transform(X, y)
    assert X_sel.shape == (100, 50)


def test_anova_selector_transform_test(XY):
    X, y = XY
    sel = AnovaSelector(k=50)
    sel.fit_transform(X[:80], y[:80])
    X_te = sel.transform(X[80:])
    assert X_te.shape == (20, 50)


def test_anova_selector_no_fit_raises(XY):
    X, _ = XY
    sel = AnovaSelector(k=10)
    with pytest.raises(RuntimeError):
        sel.transform(X)


def test_topvar_selector_shape(XY):
    X, y = XY
    sel = TopVariableSelector(k=30)
    X_sel = sel.fit_transform(X, y)
    assert X_sel.shape == (100, 30)


def test_early_fusion_shape():
    A = np.ones((50, 100))
    B = np.ones((50, 80))
    fused = early_fusion(A, B)
    assert fused.shape == (50, 180)


def test_early_fusion_mismatched_rows():
    A = np.ones((50, 10))
    B = np.ones((40, 10))
    with pytest.raises(ValueError):
        early_fusion(A, B)


def test_select_features_returns_correct_keys(XY):
    X, y = XY
    X_tr, X_te = X[:80], X[80:]
    result = select_features(
        X_tr, X_te, X_tr, X_te, y[:80],
        k_meth=20, k_expr=15, method="anova"
    )
    expected_keys = {"meth_train", "meth_test", "expr_train", "expr_test",
                     "multi_train", "multi_test", "meth_selector", "expr_selector"}
    assert expected_keys.issubset(result.keys())


def test_select_features_multi_shape(XY):
    X, y = XY
    result = select_features(
        X[:80], X[80:], X[:80], X[80:], y[:80],
        k_meth=20, k_expr=15, method="topvar"
    )
    assert result["multi_train"].shape[1] == 35   # 20 + 15
    assert result["multi_test"].shape[1]  == 35
