"""
feature_selection.py
====================
Feature selection methods for high-dimensional genomic data.

All selectors are fit on TRAINING data only, then applied to test data.
This prevents information leakage from the test set.

Methods available:
  - ANOVA F-test       (fast, interpretable univariate filter)
  - Variance threshold (remove near-constant features post-split)
  - Top-k variable     (select most variable features — unsupervised)
  - LASSO              (L1 regularization, embedded method)
  - Multi-omic fusion  (combine selected features across modalities)
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Univariate: ANOVA F-test
# ─────────────────────────────────────────────────────────────────────────────

class AnovaSelector:
    """
    Select top-k features by ANOVA F-test (equivalent to two-sample t-test
    for binary classification).

    Inspired by differential methylation / differential expression analysis.
    Fit on training data, then transform both train and test identically.

    Parameters
    ----------
    k : number of top features to select
    """

    def __init__(self, k: int = 500):
        self.k = k
        self._selector = SelectKBest(score_func=f_classif, k=k)
        self._is_fitted = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit on training (X, y) and return selected training features."""
        X_sel = self._selector.fit_transform(X, y)
        self._is_fitted = True
        return X_sel

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply selection mask to test/validation data."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        return self._selector.transform(X)

    @property
    def scores(self) -> np.ndarray:
        """F-statistic scores for each original feature."""
        return self._selector.scores_

    @property
    def pvalues(self) -> np.ndarray:
        """P-values corresponding to each F-statistic."""
        return self._selector.pvalues_

    @property
    def selected_mask(self) -> np.ndarray:
        """Boolean mask: True for selected features."""
        return self._selector.get_support()

    @property
    def n_selected(self) -> int:
        return self.selected_mask.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Top-k Most Variable (unsupervised, no label leakage risk)
# ─────────────────────────────────────────────────────────────────────────────

class TopVariableSelector:
    """
    Select the k features with the highest variance in the training set.
    Unsupervised — does not use labels, so leakage risk is minimal.
    Commonly used as a first-pass filter before supervised selection.

    Parameters
    ----------
    k : number of top-variance features to keep
    """

    def __init__(self, k: int = 1000):
        self.k = k
        self._selected_indices: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Compute variance on training X, select top-k indices.
        y is accepted but unused (for API consistency).
        """
        variances = np.nanvar(X, axis=0)
        self._selected_indices = np.argsort(variances)[::-1][: self.k]
        return X[:, self._selected_indices]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._selected_indices is None:
            raise RuntimeError("Call fit_transform() on training data first.")
        return X[:, self._selected_indices]

    @property
    def selected_indices(self) -> np.ndarray:
        return self._selected_indices


# ─────────────────────────────────────────────────────────────────────────────
# LASSO (L1 regularization) — embedded method
# ─────────────────────────────────────────────────────────────────────────────

class LassoSelector:
    """
    Use Lasso regression (L1 penalty) with cross-validated alpha to select
    features. Features with non-zero coefficients are retained.

    Better at handling correlated features than ANOVA (which treats each
    feature independently).

    Parameters
    ----------
    max_features : cap on selected features (None = no cap)
    cv           : number of cross-validation folds for alpha selection
    """

    def __init__(self, max_features: int | None = 500, cv: int = 5):
        self.max_features = max_features
        self.cv = cv
        self._lasso: LassoCV | None = None
        self._selected_mask: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self._lasso = LassoCV(cv=self.cv, n_jobs=-1, max_iter=5000)
        self._lasso.fit(X_scaled, y)

        self._selected_mask = np.abs(self._lasso.coef_) > 0

        if self.max_features and self._selected_mask.sum() > self.max_features:
            # If too many selected, keep top max_features by |coef|
            coef_abs = np.abs(self._lasso.coef_)
            top_idx = np.argsort(coef_abs)[::-1][: self.max_features]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_idx] = True
            self._selected_mask = mask

        n_sel = self._selected_mask.sum()
        print(f"  [LassoSelector] Selected {n_sel} features "
              f"(alpha={self._lasso.alpha_:.4f})")
        return X[:, self._selected_mask]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._selected_mask is None:
            raise RuntimeError("Call fit_transform() on training data first.")
        return X[:, self._selected_mask]

    @property
    def selected_mask(self) -> np.ndarray:
        return self._selected_mask


# ─────────────────────────────────────────────────────────────────────────────
# Multi-omic Fusion
# ─────────────────────────────────────────────────────────────────────────────

def early_fusion(
    *arrays: np.ndarray,
) -> np.ndarray:
    """
    Early fusion: horizontal concatenation of feature arrays.
    The simplest multi-omic strategy — all modalities concatenated into
    a single feature vector before model training.

    Parameters
    ----------
    *arrays : 2-D arrays of shape (n_samples, n_features_i)
              Must all have the same n_samples.

    Returns
    -------
    Concatenated array of shape (n_samples, sum(n_features_i))
    """
    n_rows = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n_rows:
            raise ValueError(
                f"All arrays must have the same number of rows. "
                f"Array 0 has {n_rows}, array {i} has {arr.shape[0]}."
            )
    fused = np.hstack(arrays)
    print(f"  [early_fusion] Fused {len(arrays)} modalities → "
          f"{fused.shape[1]} total features")
    return fused


def select_features(
    X_meth_train: np.ndarray,
    X_meth_test:  np.ndarray,
    X_expr_train: np.ndarray,
    X_expr_test:  np.ndarray,
    y_train:      np.ndarray,
    k_meth:       int = 500,
    k_expr:       int = 300,
    method:       str = "anova",
) -> dict:
    """
    Run feature selection on both modalities and return all feature matrices
    (single-omic and multi-omic) ready for model training.

    Parameters
    ----------
    X_meth_train / X_meth_test : methylation train/test arrays
    X_expr_train / X_expr_test : expression train/test arrays
    y_train                    : labels (training only)
    k_meth                     : features to select from methylation
    k_expr                     : features to select from expression
    method                     : "anova" | "topvar" | "lasso"

    Returns
    -------
    dict with keys:
      meth_train, meth_test     : selected methylation features
      expr_train, expr_test     : selected expression features
      multi_train, multi_test   : early-fused multi-omic features
      meth_selector             : fitted selector object (methylation)
      expr_selector             : fitted selector object (expression)
    """
    Selector = {
        "anova"  : AnovaSelector,
        "topvar" : TopVariableSelector,
        "lasso"  : LassoSelector,
    }.get(method)

    if Selector is None:
        raise ValueError(f"method must be 'anova', 'topvar', or 'lasso'")

    print(f"[feature_selection] Method={method}  k_meth={k_meth}  k_expr={k_expr}")

    meth_sel = Selector(k=k_meth)
    expr_sel = Selector(k=k_expr)

    X_meth_tr = meth_sel.fit_transform(X_meth_train, y_train)
    X_meth_te = meth_sel.transform(X_meth_test)

    X_expr_tr = expr_sel.fit_transform(X_expr_train, y_train)
    X_expr_te = expr_sel.transform(X_expr_test)

    X_multi_tr = early_fusion(X_meth_tr, X_expr_tr)
    X_multi_te = early_fusion(X_meth_te, X_expr_te)

    return {
        "meth_train"    : X_meth_tr,
        "meth_test"     : X_meth_te,
        "expr_train"    : X_expr_tr,
        "expr_test"     : X_expr_te,
        "multi_train"   : X_multi_tr,
        "multi_test"    : X_multi_te,
        "meth_selector" : meth_sel,
        "expr_selector" : expr_sel,
    }
