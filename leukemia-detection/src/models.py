"""
models.py
=========
Classifier definitions, cross-validation training, and model persistence.

All models are trained with 5-fold stratified cross-validation on the
training set. The best model (by CV AUC) is then re-fit on the full
training set before final evaluation.
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


# ─────────────────────────────────────────────────────────────────────────────
# Classifier Registry
# ─────────────────────────────────────────────────────────────────────────────

def get_classifiers(random_state: int = 42) -> dict:
    """
    Return a dictionary of classifier instances with sensible defaults.

    These are good starting points — use train_with_cv() to tune further
    via GridSearchCV or RandomizedSearchCV.

    Parameters
    ----------
    random_state : seed for reproducibility

    Returns
    -------
    dict : {name: sklearn estimator}
    """
    return {
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,        # needed for predict_proba / AUC
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            C=0.1,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        ),
        "MLP (Neural)": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation Training
# ─────────────────────────────────────────────────────────────────────────────

def train_with_cv(
    classifiers: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: list[str] | None = None,
) -> dict:
    """
    Train each classifier with stratified k-fold cross-validation and
    report mean ± std for each metric.

    Parameters
    ----------
    classifiers  : dict of {name: estimator} from get_classifiers()
    X_train      : training feature matrix
    y_train      : training labels
    n_splits     : number of CV folds
    random_state : seed for fold splitting
    scoring      : list of sklearn scoring strings
                   (default: ["roc_auc", "f1", "precision", "recall"])

    Returns
    -------
    cv_results : dict of {clf_name: {metric: (mean, std)}}
    """
    if scoring is None:
        scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = {}

    for name, clf in classifiers.items():
        scores = cross_validate(
            clf, X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_results[name] = {
            metric: (
                scores[f"test_{metric}"].mean(),
                scores[f"test_{metric}"].std(),
            )
            for metric in scoring
        }
        auc_mean, auc_std = cv_results[name]["roc_auc"]
        f1_mean,  f1_std  = cv_results[name]["f1"]
        print(f"  {name:22s}  CV AUC={auc_mean:.4f}±{auc_std:.4f}  "
              f"F1={f1_mean:.4f}±{f1_std:.4f}")

    return cv_results


def fit_final_models(
    classifiers: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Re-fit each classifier on the FULL training set after CV tuning.
    These are the models used for final test-set evaluation.

    Parameters
    ----------
    classifiers : dict of {name: estimator}
    X_train     : full training feature matrix
    y_train     : full training labels

    Returns
    -------
    fitted_models : dict of {name: fitted estimator}
    """
    fitted = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        fitted[name] = clf
        print(f"  Fitted: {name}")
    return fitted


def select_best_model(
    cv_results: dict,
    metric: str = "roc_auc",
) -> str:
    """
    Return the name of the classifier with the highest mean CV score
    for the given metric.

    Parameters
    ----------
    cv_results : output of train_with_cv()
    metric     : metric to rank by (e.g., "roc_auc", "f1")

    Returns
    -------
    Name of the best classifier
    """
    best_name = max(
        cv_results,
        key=lambda name: cv_results[name][metric][0],
    )
    best_score = cv_results[best_name][metric][0]
    print(f"  Best model by {metric}: {best_name} ({best_score:.4f})")
    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# Model Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str | Path) -> None:
    """
    Serialize a fitted model to disk using joblib.

    Parameters
    ----------
    model : fitted sklearn estimator
    path  : output file path (e.g., "results/models/best_model.pkl")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  [models] Saved model → {path}")


def load_model(path: str | Path):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : path to .pkl file saved by save_model()

    Returns
    -------
    Fitted sklearn estimator
    """
    model = joblib.load(path)
    print(f"  [models] Loaded model ← {path}")
    return model
