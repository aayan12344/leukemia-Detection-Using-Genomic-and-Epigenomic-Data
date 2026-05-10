"""
train.py

  # Simulated data:
  python scripts/train.py --mode simulate

  # Real data (methylation only):
  python scripts/train.py \
      --methylation data/processed/methylation_combined.csv \
      --labels      data/processed/labels_combined.csv

  # Real data (multi-omic):
  python scripts/train.py \
      --methylation data/processed/methylation_combined.csv \
      --expression  data/processed/expression_log2.csv \
      --labels      data/processed/labels_combined.csv
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

from src.data_loader       import load_simulated
from src.preprocessing     import OmicsPreprocessor, filter_missing, filter_low_variance
from src.feature_selection import AnovaSelector, early_fusion
from src.models            import get_classifiers, train_with_cv, fit_final_models, save_model
from src.evaluation        import evaluate_model, results_to_dataframe, print_classification_report
from src.visualization     import (plot_roc_curves, plot_confusion_matrix, plot_pca,
                                   plot_metrics_heatmap, save_individual_plots, DARK_BG)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["simulate", "csv"], default="csv")
    p.add_argument("--methylation", type=str,   default=None)
    p.add_argument("--expression",  type=str,   default=None)
    p.add_argument("--labels",      type=str,   default=None)
    p.add_argument("--k_meth",      type=int,   default=500)
    p.add_argument("--k_expr",      type=int,   default=300)
    p.add_argument("--method",      type=str,   default="anova",
                   choices=["anova", "topvar", "lasso"])
    p.add_argument("--test_size",   type=float, default=0.20)
    p.add_argument("--cv_folds",    type=int,   default=5)
    p.add_argument("--output",      type=str,   default="results/")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────
    multiomics = False

    if args.mode == "simulate":
        meth_df, expr_df, labels = load_simulated(random_state=args.seed)
        multiomics = True
    else:
        if not args.methylation or not args.labels:
            raise ValueError("Provide --methylation and --labels for csv mode.")
        meth_df   = pd.read_csv(args.methylation, index_col=0)
        labels_df = pd.read_csv(args.labels,      index_col=0)
        labels    = labels_df.loc[meth_df.index, "label"].values.astype(int)
        if args.expression:
            expr_df    = pd.read_csv(args.expression, index_col=0)
            multiomics = True

    # ── 2. Clean ─────────────────────────────────────────────────
    meth_df = filter_missing(meth_df, threshold=0.20)
    meth_df = filter_low_variance(meth_df, threshold=0.001)
    if args.mode == "csv":
        labels = labels_df.loc[meth_df.index, "label"].values.astype(int)
    if multiomics and args.expression:
        expr_df = filter_missing(expr_df,  threshold=0.05)
        expr_df = filter_low_variance(expr_df, threshold=0.01)

    # ── 3. Split ─────────────────────────────────────────────────
    X_meth = meth_df.values
    y      = labels

    if multiomics and args.expression:
        X_meth_tr, X_meth_te, X_expr_tr, X_expr_te, y_train, y_test = train_test_split(
            X_meth, expr_df.values, y,
            test_size=args.test_size, random_state=args.seed, stratify=y,
        )
    else:
        X_meth_tr, X_meth_te, y_train, y_test = train_test_split(
            X_meth, y,
            test_size=args.test_size, random_state=args.seed, stratify=y,
        )
        X_expr_tr = X_expr_te = None

    # ── 4. Preprocess ─────────────────────────────────────────────
    meth_prep = OmicsPreprocessor(impute_strategy="median", scaler="standard")
    X_meth_tr = meth_prep.fit_transform(X_meth_tr)
    X_meth_te = meth_prep.transform(X_meth_te)

    if X_expr_tr is not None:
        expr_prep = OmicsPreprocessor(impute_strategy="mean", scaler="standard")
        X_expr_tr = expr_prep.fit_transform(X_expr_tr)
        X_expr_te = expr_prep.transform(X_expr_te)

    # ── 5. Feature selection ──────────────────────────────────────
    sel  = AnovaSelector(k=min(args.k_meth, X_meth_tr.shape[1]))
    X_tr = sel.fit_transform(X_meth_tr, y_train)
    X_te = sel.transform(X_meth_te)

    if X_expr_tr is not None:
        sel_e         = AnovaSelector(k=min(args.k_expr, X_expr_tr.shape[1]))
        X_expr_tr_sel = sel_e.fit_transform(X_expr_tr, y_train)
        X_expr_te_sel = sel_e.transform(X_expr_te)
        feature_sets  = {
            "Methylation only": (X_tr,            X_te),
            "Expression only" : (X_expr_tr_sel,   X_expr_te_sel),
            "Multi-omic"      : (early_fusion(X_tr, X_expr_tr_sel),
                                 early_fusion(X_te, X_expr_te_sel)),
        }
    else:
        feature_sets = {"Methylation": (X_tr, X_te)}

    # ── 6. Train ──────────────────────────────────────────────────
    all_results    = []
    fitted_by_fset = {}

    for fset_name, (X_ftr, X_fte) in feature_sets.items():
        clfs   = get_classifiers(random_state=args.seed)
        train_with_cv(clfs, X_ftr, y_train, n_splits=args.cv_folds,
                      random_state=args.seed)
        fitted = fit_final_models(clfs, X_ftr, y_train)
        fitted_by_fset[fset_name] = fitted
        for clf_name, clf in fitted.items():
            res = evaluate_model(clf, X_fte, y_test,
                                 model_name=clf_name, verbose=False)
            res["feature_set"] = fset_name
            res["classifier"]  = clf_name
            all_results.append(res)

    # ── 7. Evaluate ───────────────────────────────────────────────
    results_df = results_to_dataframe(all_results)
    best_res   = max(all_results, key=lambda r: r["auc"])
    print_classification_report(y_test, best_res["y_pred"])
    save_model(fitted_by_fset[best_res["feature_set"]][best_res["classifier"]],
               output_dir / "models" / "best_model.pkl")

    # ── 8. Figures ────────────────────────────────────────────────
    fset_name        = list(feature_sets.keys())[0]
    X_tr_plot, X_te_plot = feature_sets[fset_name]
    X_all = np.vstack([X_tr_plot, X_te_plot])
    y_all = np.concatenate([y_train, y_test])

    # Dashboard
    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                            left=0.06, right=0.97, top=0.91, bottom=0.07)
    plot_roc_curves(all_results, y_test, feature_set=fset_name,
                    ax=fig.add_subplot(gs[0, 0]), title=f"ROC Curves — {fset_name}")
    plot_confusion_matrix(best_res["cm"],
                          title=f"Best: {best_res['classifier']}\n"
                                f"AUC={best_res['auc']:.3f}  F1={best_res['f1']:.3f}",
                          ax=fig.add_subplot(gs[0, 1]))
    plot_pca(X_all, y_all, title=f"PCA — {fset_name}", ax=fig.add_subplot(gs[0, 2]))
    plot_metrics_heatmap(results_df, feature_set=fset_name, ax=fig.add_subplot(gs[1, :]))
    fig.text(0.5, 0.965, "Leukemia Detection — AML Methylation",
             ha="center", color="white", fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "figures" / "dashboard.png",
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

    # Individual plots
    save_individual_plots(
        results     = all_results,
        results_df  = results_df,
        y_test      = y_test,
        X_train     = X_tr_plot,
        y_train     = y_train,
        X_all       = X_all,
        y_all       = y_all,
        feature_set = fset_name,
        output_dir  = str(output_dir / "figures"),
    )


if __name__ == "__main__":
    main()