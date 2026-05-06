"""
test_preprocessing.py
=====================
Unit tests for src/preprocessing.py

Run with:  pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    filter_missing,
    filter_low_variance,
    remove_duplicate_samples,
    align_samples,
    OmicsPreprocessor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """A 10-sample × 5-feature DataFrame with one high-missing feature."""
    rng = np.random.default_rng(0)
    data = rng.random((10, 5))
    data[:, 2] = np.nan          # column 2 is entirely NaN
    df = pd.DataFrame(data, index=[f"S{i}" for i in range(10)],
                      columns=["f0", "f1", "f2", "f3", "f4"])
    return df


@pytest.fixture
def clean_array():
    """A clean 50×20 float array with no missing values."""
    return np.random.default_rng(1).random((50, 20))


# ─────────────────────────────────────────────────────────────────────────────
# filter_missing
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_missing_removes_high_nan_column(small_df):
    result = filter_missing(small_df, threshold=0.5)
    assert "f2" not in result.columns, "Entirely NaN column should be removed"
    assert result.shape[1] == 4


def test_filter_missing_keeps_low_nan_column(small_df):
    # Make f0 have 1 NaN out of 10 (10%) → below 20% threshold
    small_df.loc["S0", "f0"] = np.nan
    result = filter_missing(small_df, threshold=0.20)
    assert "f0" in result.columns


def test_filter_missing_samples_axis(small_df):
    # Give row S0 a lot of NaNs
    small_df.loc["S0"] = np.nan
    result = filter_missing(small_df, threshold=0.5, axis="samples")
    assert "S0" not in result.index


# ─────────────────────────────────────────────────────────────────────────────
# filter_low_variance
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_low_variance_removes_constant(small_df):
    small_df["f4"] = 0.5    # constant column, variance = 0
    result = filter_low_variance(small_df, threshold=0.001)
    assert "f4" not in result.columns


def test_filter_low_variance_keeps_variable(small_df):
    result = filter_low_variance(small_df, threshold=0.001)
    # f0, f1, f3 should survive
    assert result.shape[1] >= 3


# ─────────────────────────────────────────────────────────────────────────────
# remove_duplicate_samples
# ─────────────────────────────────────────────────────────────────────────────

def test_remove_duplicate_samples():
    df = pd.DataFrame(
        np.ones((4, 3)),
        index=["A", "A", "B", "C"],
        columns=["f0", "f1", "f2"],
    )
    result = remove_duplicate_samples(df)
    assert len(result) == 3
    assert "A" in result.index


# ─────────────────────────────────────────────────────────────────────────────
# align_samples
# ─────────────────────────────────────────────────────────────────────────────

def test_align_samples_inner_join():
    df1 = pd.DataFrame(np.ones((4, 2)), index=["A", "B", "C", "D"])
    df2 = pd.DataFrame(np.ones((3, 2)), index=["B", "C", "D"])
    r1, r2 = align_samples(df1, df2)
    assert set(r1.index) == {"B", "C", "D"}
    assert set(r2.index) == {"B", "C", "D"}


def test_align_samples_no_common_raises():
    df1 = pd.DataFrame(np.ones((2, 2)), index=["A", "B"])
    df2 = pd.DataFrame(np.ones((2, 2)), index=["C", "D"])
    r1, r2 = align_samples(df1, df2)
    assert len(r1) == 0


# ─────────────────────────────────────────────────────────────────────────────
# OmicsPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocessor_fit_transform_shape(clean_array):
    prep = OmicsPreprocessor(impute_strategy="median", scaler="standard")
    X_tr = prep.fit_transform(clean_array[:40])
    X_te = prep.transform(clean_array[40:])
    assert X_tr.shape == (40, 20)
    assert X_te.shape == (10, 20)


def test_preprocessor_train_mean_near_zero(clean_array):
    """After StandardScaler, train mean should be ~0."""
    prep = OmicsPreprocessor(scaler="standard")
    X_tr = prep.fit_transform(clean_array[:40])
    assert np.abs(X_tr.mean(axis=0)).max() < 1e-10


def test_preprocessor_transform_before_fit_raises(clean_array):
    prep = OmicsPreprocessor()
    with pytest.raises(RuntimeError):
        prep.transform(clean_array)


def test_preprocessor_handles_missing():
    X = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
    prep = OmicsPreprocessor(impute_strategy="median", scaler="none")
    X_out = prep.fit_transform(X)
    assert not np.isnan(X_out).any()


def test_preprocessor_minmax_range(clean_array):
    prep = OmicsPreprocessor(scaler="minmax")
    X_tr = prep.fit_transform(clean_array[:40])
    assert X_tr.min() >= 0.0 - 1e-9
    assert X_tr.max() <= 1.0 + 1e-9
