"""
test_models.py
==============
Unit tests for src/models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import tempfile

from src.models import (
    get_classifiers,
    train_with_cv,
    fit_final_models,
    select_best_model,
    save_model,
    load_model,
)


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 20))
    # Make classes linearly separable for fast tests
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_get_classifiers_returns_dict():
    clfs = get_classifiers()
    assert isinstance(clfs, dict)
    assert len(clfs) > 0


def test_train_with_cv_returns_scores(simple_data):
    X, y = simple_data
    clfs = get_classifiers()
    # Only test 2 classifiers for speed
    clfs = dict(list(clfs.items())[:2])
    results = train_with_cv(clfs, X, y, n_splits=3)
    assert set(results.keys()) == set(clfs.keys())
    for name, scores in results.items():
        assert "roc_auc" in scores
        mean, std = scores["roc_auc"]
        assert 0.0 <= mean <= 1.0
        assert std >= 0.0


def test_fit_final_models(simple_data):
    X, y = simple_data
    clfs = {"LR": get_classifiers()["Logistic Regression"]}
    fitted = fit_final_models(clfs, X, y)
    assert "LR" in fitted
    # Should be able to predict
    preds = fitted["LR"].predict(X)
    assert len(preds) == len(y)


def test_select_best_model():
    cv_results = {
        "Model A": {"roc_auc": (0.92, 0.02), "f1": (0.88, 0.03)},
        "Model B": {"roc_auc": (0.95, 0.01), "f1": (0.91, 0.02)},
        "Model C": {"roc_auc": (0.89, 0.04), "f1": (0.85, 0.05)},
    }
    best = select_best_model(cv_results, metric="roc_auc")
    assert best == "Model B"


def test_save_and_load_model(simple_data):
    X, y = simple_data
    clfs = {"LR": get_classifiers()["Logistic Regression"]}
    fitted = fit_final_models(clfs, X, y)
    model = fitted["LR"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pkl"
        save_model(model, path)
        assert path.exists()

        loaded = load_model(path)
        # Predictions should match exactly
        np.testing.assert_array_equal(
            model.predict(X),
            loaded.predict(X),
        )
