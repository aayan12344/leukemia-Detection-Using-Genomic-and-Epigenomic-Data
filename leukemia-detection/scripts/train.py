"""
train.py
========
End-to-end training pipeline. Run from the project root:

  # Simulated data (for development):
  python scripts/train.py --mode simulate

  # Real CSV data:
  python scripts/train.py \
      --mode csv \
      --methylation data/processed/methylation_beta.csv \
      --expression  data/processed/expression_log2.csv \
      --labels      data/processed/labels.csv

  # Change feature selection method or number of features:
  python scripts/train.py --mode simulate --k_meth 300 --k_expr 200 --method anova

  # Save results to a custom directory:
  python scripts/train.py --mode simulate --output results/run_01/
"""

import sys
import argparse
from pathlib import Path

# Allow importing from src/ when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader       import load_simulated, load_from_csv
from src.preprocessing     import clean_dataset, OmicsPreprocessor
from src.feature_selection import select_features
from src.models            import get_classifiers, train_with_cv, fit_final_models, save_model
from src.evaluation        import evaluate_model, results_to_dataframe, print_classification_report
from src.visualization     import plot_dashboard


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Leukemia Detection — Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["simulate", "csv"], default="simulate",
                   help="Data source: simulate synthetic data or load from CSV")
    p.add_argument("--methylation", type=str, default=None,
                   help="Path to methylation CSV (required if mode=csv)")
    p.add_argument("--expression",  type=str, default=None,
                   help="Path to expression CSV  (optional — omit for methylation-only)")
    p.add_argument("--labels",      type=str, default=None,
                   help="Path to labels CSV      (required if mode=csv)")
    p.add_argument("--k_meth",   type=int, default=500,
                   help="Number of methylation features to select")
    p.add_argument("--k_expr",   type=int, default=300,
                   help="Number of expression features to select")
    p.add_argument("--method",   type=str, default="anova",
                   choices=["anova", "topvar", "lasso"],
                   help="Feature selection method")
    p.add_argument("--test_size", type=float, default=0.20,
                   help="Fraction of data held out as test set")
    p.add_argument("--cv_folds",  type=int, default=5,
                   help="Number of cross-validation folds")
    p.add_argument("--output",    type=str, default="results/",
                   help="Directory for saved figures and models")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LEUKEMIA DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────
    print("\n[1] Loading data...")
    import pandas as pd
    from src.preprocessing import filter_missing, filter_low_variance
    from src.feature_selection import AnovaSelector
    from src.visualization import (plot_roc_curves, plot_confusion_matrix,
                                   plot_pca, plot_metrics_heatmap, DARK_BG)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    multiomics = False  # flag: do we have both modalities?

    if args.mode == "simulate":
        meth_df, expr_df, labels = load_simulated(random_state=args.seed)
        multiomics = True
        print(f"    Simulated {len(labels)} samples "
              f"({labels.sum()} leukemia / {(labels==0).sum()} normal)")
    else:
        if not args.methylation or not args.labels:
            raise ValueError("For mode=csv, provide at least --methylation and --labels")
        meth_df   = pd.read_csv(args.methylation, index_col=0)
        labels_df = pd.read_csv(args.labels, index_col=0)
        labels    = labels_df.loc[meth_df.index, "label"].values.astype(int)
        print(f"    Methylation: {meth_df.shape}")
        print(f"    Labels — Leukemia: {labels.sum()}  Normal: {(labels==0).sum()}")
        if args.expression:
            expr_df   = pd.read_csv(args.expression, index_col=0)
            multiomics = True
            print(f"    Expression : {expr_df.shape}")
        else:
            print("    Expression : not provided — running methylation-only mode")

    # ── 2. Clean ─────────────────────────────────────────────────
    print("\n[2] Cleaning dataset (before split)...")
    meth_df = filter_missing(meth_df, threshold=0.20)
    meth_df = filter_low_variance(meth_df, threshold=0.001)
    labels  = labels_df.loc[meth_df.index, "label"].values.astype(int) \
              if args.mode == "csv" else labels
    if multiomics and args.mode == "csv" and args.expression:
        expr_df = filter_missing(expr_df, threshold=0.05)
        expr_df = filter_low_variance(expr_df, threshold=0.01)
    print(f"    Methylation: {meth_df.shape}")

    # ── 3. CRITICAL SPLIT ─────────────────────────────────────────
    print("\n[3] Stratified train/test split...")
    X_meth = meth_df.values
    y      = labels

    if multiomics and args.expression:
        (X_meth_tr, X_meth_te,
         X_expr_tr, X_expr_te,
         y_train,   y_test) = train_test_split(
            X_meth, expr_df.values, y,
            test_size=args.test_size, random_state=args.seed, stratify=y,
        )
    else:
        X_meth_tr, X_meth_te, y_train, y_test = train_test_split(
            X_meth, y,
            test_size=args.test_size, random_state=args.seed, stratify=y,
        )
        X_expr_tr = X_expr_te = None

    print(f"    Train: {len(y_train)} (AML={y_train.sum()}, Normal={(y_train==0).sum()})")
    print(f"    Test : {len(y_test)}  (AML={y_test.sum()},  Normal={(y_test==0).sum()})")

    # ── 4. Preprocessing ─────────────────────────────────────────
    print("\n[4] Preprocessing (fit on train, apply to test)...")
    meth_prep = OmicsPreprocessor(impute_strategy="median", scaler="standard")
    X_meth_tr = meth_prep.fit_transform(X_meth_tr)
    X_meth_te = meth_prep.transform(X_meth_te)

    if X_expr_tr is not None:
        expr_prep = OmicsPreprocessor(impute_strategy="mean", scaler="standard")
        X_expr_tr = expr_prep.fit_transform(X_expr_tr)
        X_expr_te = expr_prep.transform(X_expr_te)

    # ── 5. Feature selection ─────────────────────────────────────
    print(f"\n[5] Feature selection (method={args.method})...")
    k = min(args.k_meth, X_meth_tr.shape[1])
    sel = AnovaSelector(k=k)
    X_tr = sel.fit_transform(X_meth_tr, y_train)
    X_te = sel.transform(X_meth_te)
    print(f"    Selected {X_tr.shape[1]} methylation features")

    if X_expr_tr is not None:
        k_e = min(args.k_expr, X_expr_tr.shape[1])
        sel_e = AnovaSelector(k=k_e)
        X_expr_tr_sel = sel_e.fit_transform(X_expr_tr, y_train)
        X_expr_te_sel = sel_e.transform(X_expr_te)
        from src.feature_selection import early_fusion
        X_tr_multi = early_fusion(X_tr, X_expr_tr_sel)
        X_te_multi = early_fusion(X_te, X_expr_te_sel)
        feature_sets = {
            "Methylation only": (X_tr, X_te),
            "Expression only" : (X_expr_tr_sel, X_expr_te_sel),
            "Multi-omic"      : (X_tr_multi, X_te_multi),
        }
    else:
        feature_sets = {"Methylation": (X_tr, X_te)}

    # ── 6. Train with CV ─────────────────────────────────────────
    print("\n[6] Cross-validation training...")
    all_results   = []
    fitted_by_fset = {}

    for fset_name, (X_ftr, X_fte) in feature_sets.items():
        print(f"\n  ── {fset_name} ──")
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
    print("\n[7] Final test set evaluation...")
    results_df = results_to_dataframe(all_results)
    print(results_df.to_string(index=False))

    best_res = max(all_results, key=lambda r: r["auc"])
    print(f"\n  ★ Best: {best_res['classifier']} [{best_res['feature_set']}]"
          f"  AUC={best_res['auc']:.4f}  F1={best_res['f1']:.4f}")
    print_classification_report(y_test, best_res["y_pred"])

    # ── 8. Save best model ────────────────────────────────────────
    best_clf = fitted_by_fset[best_res["feature_set"]][best_res["classifier"]]
    save_model(best_clf, output_dir / "models" / "best_model.pkl")

    # ── 9. Dashboard ─────────────────────────────────────────────
    print("\n[8] Generating results dashboard...")
    fset_name = list(feature_sets.keys())[0]
    X_tr_plot, X_te_plot = feature_sets[fset_name]

    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                            left=0.06, right=0.97, top=0.91, bottom=0.07)

    plot_roc_curves(all_results, y_test, feature_set=fset_name,
                    ax=fig.add_subplot(gs[0, 0]),
                    title=f"ROC Curves — {fset_name}")
    plot_confusion_matrix(best_res["cm"],
                          title=f"Best: {best_res['classifier']}\n"
                                f"AUC={best_res['auc']:.3f}  F1={best_res['f1']:.3f}",
                          ax=fig.add_subplot(gs[0, 1]))

    X_all = np.vstack([X_tr_plot, X_te_plot])
    y_all = np.concatenate([y_train, y_test])
    plot_pca(X_all, y_all, title=f"PCA — {fset_name}",
             ax=fig.add_subplot(gs[0, 2]))
    plot_metrics_heatmap(results_df, feature_set=fset_name,
                         ax=fig.add_subplot(gs[1, :]))

    fig.text(0.5, 0.965,
             "Leukemia Detection — Real AML Methylation Data",
             ha="center", color="white", fontsize=13, fontweight="bold")

    dash_path = str(output_dir / "figures" / "dashboard.png")
    fig.savefig(dash_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    Saved → {dash_path}")

    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  Figures → {output_dir}/figures/")
    print(f"  Models  → {output_dir}/models/")
    print("=" * 60)



if __name__ == "__main__":
    main()