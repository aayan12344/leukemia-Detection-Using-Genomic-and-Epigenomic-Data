"""
visualization.py
================
All plotting functions for the leukemia detection pipeline.

Each function is self-contained: pass in the data, get back a figure.
Call fig.savefig(path) to save, or plt.show() to display interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve

# ─── Color palette ────────────────────────────────────────────────────────────
CLASSIFIER_COLORS = {
    "SVM (RBF)"         : "#E63946",
    "Random Forest"     : "#2A9D8F",
    "Gradient Boosting" : "#F4A261",
    "Logistic Regression": "#457B9D",
    "MLP (Neural)"      : "#9B5DE5",
}

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT_COLOR = "#cccccc"
GRID_COLOR = "#222222"


def _dark_ax(ax):
    """Apply dark theme to an axes."""
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.tick_params(colors="#888888", labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: list[dict],
    y_test: np.ndarray,
    feature_set: str = "Multi-omic",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot ROC curves for all classifiers on a given feature set.

    Parameters
    ----------
    results     : list of result dicts from evaluation.evaluate_model()
    y_test      : true labels
    feature_set : which feature set to plot (filter from results)
    ax          : existing axes to draw on (creates new figure if None)
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    _dark_ax(ax)

    ax.plot([0, 1], [0, 1], ":", color="#555555", lw=1, label="Random")

    for r in results:
        if r.get("feature_set", "") != feature_set:
            continue
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        color = CLASSIFIER_COLORS.get(r["classifier"], "#aaaaaa")
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{r['classifier']} (AUC={r['auc']:.3f})")

    ax.set_title(title or f"ROC Curves — {feature_set}",
                 color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("True Positive Rate",  color="#aaaaaa", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.15, labelcolor="white",
              facecolor=DARK_BG, edgecolor="#333333")
    return fig or ax.figure


def plot_pr_curves(
    results: list[dict],
    y_test: np.ndarray,
    feature_set: str = "Multi-omic",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot Precision-Recall curves."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    _dark_ax(ax)

    baseline = y_test.mean()
    ax.axhline(baseline, color="#555555", ls=":", lw=1,
               label=f"Baseline ({baseline:.2f})")

    for r in results:
        if r.get("feature_set", "") != feature_set:
            continue
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        color = CLASSIFIER_COLORS.get(r["classifier"], "#aaaaaa")
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{r['classifier']} (AP={r['avg_precision']:.3f})")

    ax.set_title(f"Precision-Recall — {feature_set}",
                 color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.set_xlabel("Recall",    color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Precision", color="#aaaaaa", fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.15, labelcolor="white",
              facecolor=DARK_BG, edgecolor="#333333")
    return fig or ax.figure


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a single confusion matrix as a heatmap."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor=DARK_BG)
    _dark_ax(ax)

    cmap = LinearSegmentedColormap.from_list("leu", [DARK_BG, "#E63946"])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, ax=ax,
        xticklabels=["Normal", "Leukemia"],
        yticklabels=["Normal", "Leukemia"],
        cbar=False,
        annot_kws={"size": 14, "color": "white", "weight": "bold"},
        linewidths=1, linecolor="#333333",
    )
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.set_xlabel("Predicted", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Actual",    color="#aaaaaa", fontsize=9)
    return fig or ax.figure


def plot_auc_comparison(
    results_df,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Bar chart comparing AUC across classifiers and feature sets."""
    import pandas as pd
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    _dark_ax(ax)

    pivot = results_df.pivot(
        index="Classifier", columns="Feature Set", values="AUC-ROC"
    )
    fset_colors = ["#264653", "#2A9D8F", "#E9C46A"]
    x = np.arange(len(pivot.index))
    w = 0.26

    for i, (col, color) in enumerate(zip(pivot.columns, fset_colors)):
        ax.bar(x + i * w, pivot[col], w, label=col,
               color=color, edgecolor="#111111", linewidth=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels(pivot.index, color=TEXT_COLOR, fontsize=9, rotation=15)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("AUC-ROC", color="#aaaaaa", fontsize=9)
    ax.set_title("AUC-ROC by Classifier & Feature Set",
                 color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.15, labelcolor="white",
              facecolor=DARK_BG, edgecolor="#333333")
    ax.yaxis.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    return fig or ax.figure


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "PCA — Feature Space",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """2-D PCA scatter coloured by class label."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=DARK_BG)
    _dark_ax(ax)

    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)

    colors = ["#2A9D8F" if yi == 0 else "#E63946" for yi in y]
    ax.scatter(X2[:, 0], X2[:, 1], c=colors, alpha=0.65, s=18,
               edgecolors="none")

    legend_elems = [
        Patch(facecolor="#2A9D8F", label="Normal"),
        Patch(facecolor="#E63946", label="Leukemia"),
    ]
    ax.legend(handles=legend_elems, fontsize=8.5, framealpha=0.15,
              labelcolor="white", facecolor=DARK_BG, edgecolor="#333333")
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        color="#aaaaaa", fontsize=9,
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        color="#aaaaaa", fontsize=9,
    )
    return fig or ax.figure


def plot_methylation_heatmap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selector,
    n_top: int = 30,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Heatmap of the top-n most differentially methylated CpG probes.

    Parameters
    ----------
    X_train  : training methylation matrix (post-preprocessing, pre-selection)
    y_train  : training labels
    selector : fitted AnovaSelector with .scores attribute
    n_top    : number of top probes to display
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
    _dark_ax(ax)

    # scores are over original features; heatmap uses already-selected matrix
    # so just show top n_top columns of the (already selected) X_train
    n_show = min(n_top, X_train.shape[1])
    top_idx = np.arange(n_show)
    hm_data = X_train[:, top_idx]
    order   = np.argsort(y_train)

    im = ax.imshow(
        hm_data[order].T, aspect="auto",
        cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
    )
    ax.set_title(f"Top {n_top} CpG Probes (sorted by label)",
                 color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.set_xlabel("Samples",    color="#aaaaaa", fontsize=9)
    ax.set_ylabel("CpG Probes", color="#aaaaaa", fontsize=9)

    cbar = (fig or ax.figure).colorbar(im, ax=ax, fraction=0.04)
    cbar.ax.tick_params(colors="#888888", labelsize=7)
    cbar.set_label("Z-score", color="#aaaaaa", fontsize=8)
    return fig or ax.figure


def plot_metrics_heatmap(
    results_df,
    feature_set: str = "Multi-omic",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Heatmap of all metrics for one feature set across classifiers."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=DARK_BG)
    _dark_ax(ax)

    sub = results_df[results_df["Feature Set"] == feature_set].set_index("Classifier")
    cols = ["AUC-ROC", "F1", "Precision", "Recall", "Accuracy"]
    sub  = sub[cols]

    cmap2 = LinearSegmentedColormap.from_list(
        "met", ["#0d3b52", "#2A9D8F", "#E9C46A"]
    )
    sns.heatmap(
        sub, annot=True, fmt=".3f", cmap=cmap2, ax=ax,
        cbar=False,
        annot_kws={"size": 9, "color": "black", "weight": "bold"},
        linewidths=0.5, linecolor=DARK_BG,
        vmin=0.7, vmax=1.0,
    )
    ax.set_title(f"Metrics Heatmap — {feature_set}",
                 color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    return fig or ax.figure


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: All plots in one figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(
    results: list[dict],
    results_df,
    y_test: np.ndarray,
    X_meth_train: np.ndarray,
    y_train: np.ndarray,
    meth_selector,
    X_multi_all: np.ndarray,
    y_all: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Full 8-panel results dashboard.

    Parameters
    ----------
    results       : list of result dicts from evaluation.evaluate_all()
    results_df    : DataFrame from evaluation.results_to_dataframe()
    y_test        : test labels
    X_meth_train  : training methylation matrix (for heatmap)
    y_train       : training labels
    meth_selector : fitted methylation AnovaSelector
    X_multi_all   : multi-omic matrix (train + test combined, for PCA)
    y_all         : labels for all samples (train + test combined)
    save_path     : if provided, save figure to this path
    """
    fig = plt.figure(figsize=(22, 18), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )

    best_row = max(results, key=lambda r: r["auc"])

    plot_roc_curves(results, y_test, feature_set="Multi-omic",
                    ax=fig.add_subplot(gs[0, 0]))
    plot_pr_curves(results, y_test, feature_set="Multi-omic",
                   ax=fig.add_subplot(gs[0, 1]))
    plot_confusion_matrix(
        best_row["cm"],
        title=f"Confusion Matrix\n{best_row['classifier']} ({best_row['feature_set']})",
        ax=fig.add_subplot(gs[0, 2]),
    )
    plot_auc_comparison(results_df, ax=fig.add_subplot(gs[1, :2]))
    plot_pca(X_multi_all, y_all, title="PCA — Multi-omic Feature Space",
             ax=fig.add_subplot(gs[2, 0]))
    plot_methylation_heatmap(X_meth_train, y_train, meth_selector,
                             ax=fig.add_subplot(gs[2, 1]))
    plot_metrics_heatmap(results_df, feature_set="Multi-omic",
                         ax=fig.add_subplot(gs[2, 2]))

    fig.text(0.5, 0.97,
             "Leukemia Detection — Genomic & Epigenomic ML Pipeline",
             ha="center", va="top", color="white",
             fontsize=15, fontweight="bold")
    fig.text(0.5, 0.955,
             "DNA Methylation + Gene Expression  |  Leukemia vs Normal  |  Binary Classification",
             ha="center", va="top", color="#888888", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  [visualization] Saved dashboard → {save_path}")

    return fig
