"""
evaluation.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)



def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True,
) -> dict:
    """
    Compute all classification metrics for one fitted model on the test set.

    Parameters
    ----------
    model      : fitted sklearn estimator with predict_proba()
    X_test     : test feature matrix (preprocessed, features selected)
    y_test     : true test labels
    model_name : display name for reporting
    verbose    : print metrics to stdout

    Returns
    -------
    dict with keys: name, auc, f1, precision, recall, accuracy,
                    avg_precision, y_pred, y_prob, cm
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "name"          : model_name,
        "auc"           : roc_auc_score(y_test, y_prob),
        "f1"            : f1_score(y_test, y_pred),
        "precision"     : precision_score(y_test, y_pred),
        "recall"        : recall_score(y_test, y_pred),
        "accuracy"      : accuracy_score(y_test, y_pred),
        "avg_precision" : average_precision_score(y_test, y_prob),
        "y_pred"        : y_pred,
        "y_prob"        : y_prob,
        "cm"            : confusion_matrix(y_test, y_pred),
    }

    return result


def evaluate_all(
    fitted_models: dict,
    feature_sets: dict,
    y_test: np.ndarray,
    verbose: bool = True,
) -> list[dict]:
    """
    Evaluate all classifiers across all feature sets (methylation, expression,
    multi-omic) and return a list of result dicts.

    Parameters
    ----------
    fitted_models : {clf_name: {fset_name: fitted_model}}
                    or {clf_name: fitted_model} (single feature set)
    feature_sets  : {fset_name: (X_train, X_test)} — only X_test is used
    y_test        : true test labels

    Returns
    -------
    List of result dicts from evaluate_model(), one per (clf, fset) pair
    """
    results = []
    for fset_name, (_, X_te) in feature_sets.items():
        for clf_name, clf in fitted_models[fset_name].items():
            label = f"{clf_name} [{fset_name}]"
            res = evaluate_model(clf, X_te, y_test,
                                 model_name=label, verbose=verbose)
            res["feature_set"] = fset_name
            res["classifier"]  = clf_name
            results.append(res)
    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of result dicts to a tidy DataFrame for easy comparison.

    Parameters
    ----------
    results : list of dicts from evaluate_model() or evaluate_all()

    Returns
    -------
    DataFrame with one row per (classifier, feature_set)
    """
    rows = []
    for r in results:
        rows.append({
            "Classifier"   : r.get("classifier", r["name"]),
            "Feature Set"  : r.get("feature_set", "—"),
            "AUC-ROC"      : round(r["auc"],           4),
            "F1"           : round(r["f1"],            4),
            "Precision"    : round(r["precision"],     4),
            "Recall"       : round(r["recall"],        4),
            "Accuracy"     : round(r["accuracy"],      4),
            "Avg Precision": round(r["avg_precision"], 4),
        })
    df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    return df


def get_roc_curve(y_test: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_test, y_prob)


def get_pr_curve(y_test: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Return (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_test, y_prob)


def print_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
) -> None:
    """Print full sklearn classification report."""
    if target_names is None:
        target_names = ["Normal", "Leukemia"]