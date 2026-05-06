"""
data_loader.py
==============
Functions for loading and simulating leukemia genomic/epigenomic data.

Supported sources:
  - Simulated data (for development & testing)
  - Local CSV files (processed TCGA / GEO exports)
  - GEO download via GEOparse (optional dependency)

Real data sources:
  TCGA-LAML / TARGET-ALL  →  GDC Data Portal (https://portal.gdc.cancer.gov)
  GSE68978                →  AML DNA methylation 450K
  GSE13159                →  Large leukemia gene expression compendium
  GSE7186                 →  Normal bone marrow controls
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Data
# ─────────────────────────────────────────────────────────────────────────────

def simulate_methylation(
    n_leukemia: int = 150,
    n_normal:   int = 100,
    n_cpg:      int = 5000,
    n_informative: int = 500,
    missing_rate: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate DNA methylation beta values (range 0–1).

    Leukemia biology modelled:
      • Focal hypermethylation at informative CpG islands (TSS regions)
        → beta ~ Beta(8, 2) for leukemia vs Beta(2, 8) for normal
      • Global hypomethylation background (hallmark of cancer)
      • ~missing_rate fraction of values set to NaN (realistic dropout)

    Parameters
    ----------
    n_leukemia    : number of leukemia samples (label = 1)
    n_normal      : number of normal samples   (label = 0)
    n_cpg         : total number of CpG probes
    n_informative : CpGs that are differentially methylated
    missing_rate  : fraction of values to set as NaN
    random_state  : reproducibility seed

    Returns
    -------
    df     : DataFrame (samples × CpGs), beta values in [0, 1]
    labels : 1-D array of int labels (1=leukemia, 0=normal)
    """
    rng = np.random.default_rng(random_state)
    n_total = n_leukemia + n_normal
    labels  = np.array([1] * n_leukemia + [0] * n_normal)

    # Differentially methylated CpGs
    beta_leu  = rng.beta(8, 2, (n_leukemia, n_informative))   # hypermethylated
    beta_norm = rng.beta(2, 8, (n_normal,   n_informative))   # hypomethylated
    info_block = np.vstack([beta_leu, beta_norm])

    # Background CpGs
    bg_block = rng.beta(2, 2, (n_total, n_cpg - n_informative))

    beta = np.hstack([info_block, bg_block])

    # Inject missing values
    mask = rng.random(beta.shape) < missing_rate
    beta[mask] = np.nan

    cpg_ids    = [f"cg{str(i).zfill(8)}" for i in range(n_cpg)]
    sample_ids = [f"METH_{i:04d}" for i in range(n_total)]
    df = pd.DataFrame(beta, index=sample_ids, columns=cpg_ids)
    return df, labels


def simulate_expression(
    sample_ids: list[str],
    n_leukemia: int = 150,
    n_genes:    int = 3000,
    n_informative: int = 300,
    missing_rate: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate log2-normalised gene expression values.

    Biology modelled:
      • Upregulation of HOXA/HOXB cluster and FLT3 targets in leukemia
        → expr ~ N(8, 1.5) for leukemia vs N(5, 1.5) for normal
      • Background genes similar across groups
      • ~missing_rate dropout (zeros → NaN)

    Parameters
    ----------
    sample_ids    : list of sample IDs (must match methylation DataFrame index)
    n_leukemia    : how many of those samples are leukemia (first N rows)
    n_genes       : total number of genes
    n_informative : genes differentially expressed
    missing_rate  : fraction of values to set as NaN
    random_state  : reproducibility seed

    Returns
    -------
    df : DataFrame (samples × genes), log2-normalised expression
    """
    rng = np.random.default_rng(random_state + 1)
    n_total = len(sample_ids)
    n_normal = n_total - n_leukemia

    expr_leu  = rng.normal(8.0, 1.5, (n_leukemia, n_informative))
    expr_norm = rng.normal(5.0, 1.5, (n_normal,   n_informative))
    info_block = np.vstack([expr_leu, expr_norm])

    bg_block = rng.normal(6.5, 2.0, (n_total, n_genes - n_informative))
    expr = np.hstack([info_block, bg_block])

    mask = rng.random(expr.shape) < missing_rate
    expr[mask] = np.nan

    gene_ids = [f"GENE_{i:05d}" for i in range(n_genes)]
    df = pd.DataFrame(expr, index=sample_ids, columns=gene_ids)
    return df


def load_simulated(
    n_leukemia: int = 150,
    n_normal:   int = 100,
    n_cpg:      int = 5000,
    n_genes:    int = 3000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Convenience wrapper: returns (methylation_df, expression_df, labels).
    Drop-in replacement for load_from_csv() during development.
    """
    meth_df, labels = simulate_methylation(
        n_leukemia=n_leukemia, n_normal=n_normal,
        n_cpg=n_cpg, random_state=random_state,
    )
    expr_df = simulate_expression(
        sample_ids=list(meth_df.index),
        n_leukemia=n_leukemia,
        n_genes=n_genes,
        random_state=random_state,
    )
    return meth_df, expr_df, labels


# ─────────────────────────────────────────────────────────────────────────────
# Real Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(
    methylation_path: str | Path,
    expression_path:  str | Path,
    labels_path:      str | Path,
    label_column:     str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load pre-processed data from CSV files.

    Expected format
    ---------------
    methylation_path : samples × CpGs, index = sample IDs, values in [0, 1]
    expression_path  : samples × genes, index = sample IDs, log2-normalised
    labels_path      : index = sample IDs, column `label_column` = 0 or 1

    Returns
    -------
    meth_df  : DataFrame (samples × CpGs)
    expr_df  : DataFrame (samples × genes)
    labels   : 1-D int array aligned to common samples
    """
    meth_df   = pd.read_csv(methylation_path, index_col=0)
    expr_df   = pd.read_csv(expression_path,  index_col=0)
    labels_df = pd.read_csv(labels_path,      index_col=0)

    # Align on common samples
    common = meth_df.index.intersection(expr_df.index).intersection(labels_df.index)
    if len(common) == 0:
        raise ValueError("No common samples found across the three files.")

    meth_df   = meth_df.loc[common]
    expr_df   = expr_df.loc[common]
    labels    = labels_df.loc[common, label_column].values.astype(int)

    print(f"[data_loader] Loaded {len(common)} samples from CSV files.")
    return meth_df, expr_df, labels


def load_from_geo(accession: str, destdir: str = "./data/raw/") -> pd.DataFrame:
    """
    Download and parse a GEO dataset using GEOparse.

    Requires:  pip install GEOparse

    Parameters
    ----------
    accession : GEO accession, e.g. "GSE68978"
    destdir   : directory to save downloaded files

    Returns
    -------
    DataFrame of GSM sample values (features × samples)

    Example
    -------
    >>> meth_df = load_from_geo("GSE68978")
    """
    try:
        import GEOparse
    except ImportError:
        raise ImportError(
            "GEOparse is required for GEO downloads.\n"
            "Install it with:  pip install GEOparse"
        )

    Path(destdir).mkdir(parents=True, exist_ok=True)
    gse = GEOparse.get_GEO(accession, destdir=destdir, silent=True)

    # Pivot all GSM samples into a single matrix
    df = gse.pivot_samples("VALUE")
    print(f"[data_loader] Loaded {accession}: {df.shape} (features × samples)")
    return df
