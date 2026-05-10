"""
feature_selection.py

"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler




class AnovaSelector:

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



class TopVariableSelector:

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



class LassoSelector:

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
        return X[:, self._selected_mask]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._selected_mask is None:
            raise RuntimeError("Call fit_transform() on training data first.")
        return X[:, self._selected_mask]

    @property
    def selected_mask(self) -> np.ndarray:
        return self._selected_mask



def early_fusion(
    *arrays: np.ndarray,
) -> np.ndarray:

    n_rows = arrays[0].shape[0]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != n_rows:
            raise ValueError(
                f"All arrays must have the same number of rows. "
                f"Array 0 has {n_rows}, array {i} has {arr.shape[0]}."
            )
    fused = np.hstack(arrays)
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
    
    Selector = {
        "anova"  : AnovaSelector,
        "topvar" : TopVariableSelector,
        "lasso"  : LassoSelector,
    }.get(method)

    if Selector is None:
        raise ValueError(f"method must be 'anova', 'topvar', or 'lasso'")


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