"""
preprocessing.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def filter_missing(
    df: pd.DataFrame,
    threshold: float = 0.20,
    axis: str = "features",
) -> pd.DataFrame:
    """
    Remove features (columns) or samples (rows) with too many missing values.

    Parameters
    ----------
    df        : input DataFrame
    threshold : maximum allowed fraction of missing values (0–1)
    axis      : "features" (columns) or "samples" (rows)

    Returns
    -------
    Filtered DataFrame
    """
    if axis == "features":
        miss_rate = df.isna().mean(axis=0)
        keep = miss_rate[miss_rate <= threshold].index
        return df[keep]
    elif axis == "samples":
        miss_rate = df.isna().mean(axis=1)
        keep = miss_rate[miss_rate <= threshold].index
        return df.loc[keep]
    else:
        raise ValueError("axis must be 'features' or 'samples'")


def filter_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Remove features with near-zero variance — they carry no discriminative
    information and add noise to downstream models.

    Parameters
    ----------
    df        : input DataFrame (NaNs handled via skipna)
    threshold : minimum variance to keep a feature

    Returns
    -------
    Filtered DataFrame
    """
    variances = df.var(axis=0, skipna=True)
    keep = variances[variances > threshold].index
    return df[keep]


def remove_duplicate_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows (same sample measured twice).
    """
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    return df


def align_samples(
    *dfs: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """
    Align multiple DataFrames to their common sample index (inner join).
    Useful when methylation and expression come from different pipelines
    and may not have identical sample sets.

    Parameters
    ----------
    *dfs : any number of DataFrames with sample IDs as index

    Returns
    -------
    Tuple of DataFrames, all with identical index
    """
    common = dfs[0].index
    for df in dfs[1:]:
        common = common.intersection(df.index)
    n_common = len(common)
    return tuple(df.loc[common] for df in dfs)


def clean_dataset(
    meth_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    labels:  np.ndarray,
    meth_missing_threshold: float = 0.20,
    expr_missing_threshold: float = 0.05,
    meth_var_threshold:     float = 0.001,
    expr_var_threshold:     float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Full cleaning pipeline applied to the ENTIRE dataset before splitting.

    Steps:
      1. Remove high-missingness features
      2. Remove near-zero-variance features
      3. Remove duplicate samples
      4. Align samples across modalities

    Parameters
    ----------
    meth_df : methylation DataFrame (samples × CpGs)
    expr_df : expression DataFrame  (samples × genes)
    labels  : int array of labels aligned to meth_df rows

    Returns
    -------
    Cleaned (meth_df, expr_df, labels)
    """

    meth_df = filter_missing(meth_df, threshold=meth_missing_threshold)
    meth_df = filter_low_variance(meth_df, threshold=meth_var_threshold)
    meth_df = remove_duplicate_samples(meth_df)

    expr_df = filter_missing(expr_df, threshold=expr_missing_threshold)
    expr_df = filter_low_variance(expr_df, threshold=expr_var_threshold)
    expr_df = remove_duplicate_samples(expr_df)

    meth_df, expr_df = align_samples(meth_df, expr_df)
    labels = labels[:len(meth_df)]

    return meth_df, expr_df, labels



class OmicsPreprocessor:
    """
    Handles imputation and scaling for one omic modality.

    Usage
    -----
    prep = OmicsPreprocessor(impute_strategy="median", scaler="standard")
    X_train = prep.fit_transform(X_train)   # fit on training data
    X_test  = prep.transform(X_test)        # apply same params to test
    """

    VALID_SCALERS = ("standard", "minmax", "none")

    def __init__(
        self,
        impute_strategy: str = "median",
        scaler: str = "standard",
    ):
        """
        Parameters
        ----------
        impute_strategy : "median", "mean", or "most_frequent"
        scaler          : "standard" (Z-score), "minmax" (0–1), or "none"
        """
        if scaler not in self.VALID_SCALERS:
            raise ValueError(f"scaler must be one of {self.VALID_SCALERS}")

        self._imputer = SimpleImputer(strategy=impute_strategy)
        self._scaler_type = scaler
        self._scaler = (
            StandardScaler() if scaler == "standard" else
            MinMaxScaler()   if scaler == "minmax"   else
            None
        )
        self._is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X (training data) and return transformed X."""
        X = self._imputer.fit_transform(X)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        self._is_fitted = True
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply already-fitted transforms to X (test/validation data)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() on training data first.")
        X = self._imputer.transform(X)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return X

    @property
    def train_mean(self) -> np.ndarray | None:
        """Mean vector used for Z-score scaling (training data only)."""
        if isinstance(self._scaler, StandardScaler) and self._is_fitted:
            return self._scaler.mean_
        return None

    @property
    def train_std(self) -> np.ndarray | None:
        """Std vector used for Z-score scaling (training data only)."""
        if isinstance(self._scaler, StandardScaler) and self._is_fitted:
            return self._scaler.scale_
        return None