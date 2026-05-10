# Leukemia Detection via DNA Methylation

Binary classification of AML leukemia vs. normal bone marrow using DNA methylation data and machine learning.

**Dataset:** 146 samples (106 AML · 40 Normal) — GSE58477 + GSE63409 from NCBI GEO  
**Best result:** AUC-ROC = 1.0 · F1 = 0.927 · Accuracy = 90%

---

## Reproducing Our Results

### Step 1 — Clone the repo

```bash
git clone https://github.com/aayan12344/eukemia-Detection-Using-Genomic-and-Epigenomic-Data.git
cd eukemia-Detection-Using-Genomic-and-Epigenomic-Data/leukemia-detection
```

### Step 2 — Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Download the raw data

The data is publicly available on NCBI GEO at no cost and requires no account.

1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58477
   - Scroll to the bottom → click **Series Matrix File(s)** → download `GSE58477_series_matrix.txt.gz`

2. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63409
   - Scroll to the bottom → click **Series Matrix File(s)** → download `GSE63409_series_matrix.txt.gz`

3. Place both files in `data/raw/`:

```
data/raw/
├── GSE58477_series_matrix.txt.gz
└── GSE63409_series_matrix.txt.gz
```

### Step 4 — Prepare the data

```bash
python scripts/prepare_data.py \
    --datasets data/raw/GSE58477_series_matrix.txt.gz \
               data/raw/GSE63409_series_matrix.txt.gz
```

This parses the raw files and saves two CSVs to `data/processed/`:
- `methylation_combined.csv` — 146 samples × 5,000 CpG features
- `labels_combined.csv` — sample labels (1 = AML, 0 = Normal)

### Step 5 — Run the pipeline

```bash
python scripts/train.py --mode csv \
    --methylation data/processed/methylation_combined.csv \
    --labels      data/processed/labels_combined.csv
```

### Step 6 — Check the results

```
results/
├── figures/
│   ├── dashboard.png
│   ├── 01_roc_curves.png
│   ├── 02_precision_recall.png
│   ├── 03_confusion_matrix.png
│   ├── 04_metrics_heatmap.png
│   ├── 05_pca.png
│   ├── 06_methylation_heatmap.png
│   └── 07_auc_comparison.png
└── models/
    └── best_model.pkl
```

---

## Project Structure

```
leukemia-detection/
├── src/
│   ├── data_loader.py        # Load GEO / simulate data
│   ├── preprocessing.py      # Imputation and scaling
│   ├── feature_selection.py  # ANOVA, variance, LASSO
│   ├── models.py             # Classifiers and CV training
│   ├── evaluation.py         # Metrics and reporting
│   └── visualization.py      # All plots
├── scripts/
│   ├── prepare_data.py       # Parse raw GEO files → clean CSVs
│   └── train.py              # Train and evaluate all models
├── data/
│   ├── raw/                  # Raw GEO downloads (not committed)
│   └── processed/            # Cleaned CSVs (not committed)
└── results/
    ├── figures/              # Generated plots
    └── models/               # Saved model files
```

---

## Pipeline

```
Raw GEO Files (.txt.gz)
        ↓
prepare_data.py — parse, align, combine datasets
        ↓
Feature Filtering — remove low-variance / high-missing CpGs
        ↓
Stratified 80/20 Train/Test Split  ← before any normalization
        ↓
Median Imputation + Z-score Scaling  ← fit on train only
        ↓
ANOVA Feature Selection — top 500 CpGs  ← train only
        ↓
5-Fold Cross-Validation Training
(SVM, Random Forest, Gradient Boosting, Logistic Regression, MLP)
        ↓
Final Evaluation on Held-Out Test Set
```

---

## Adding More Datasets

To add a new GEO methylation dataset, open `scripts/prepare_data.py` and add a rule to the `LABEL_RULES` dictionary at the top of the file:

```python
"GSExxxxx": lambda src, char: (
    0 if "normal" in src.lower()
    else 1 if "leukemia" in src.lower()
    else None
),
```

Then include it in the prepare step:

```bash
python scripts/prepare_data.py \
    --datasets data/raw/GSE58477_series_matrix.txt.gz \
               data/raw/GSE63409_series_matrix.txt.gz \
               data/raw/GSExxxxx_series_matrix.txt.gz
```

---

## References

- Hovestadt et al. (2025). Rapid epigenomic classification of acute leukemia. *Nature Genetics.* https://doi.org/10.1038/s41588-025-02321-z
- Gandomi et al. (2024). Attention-based deep learning for ALL classification. *Scientific Reports.* https://doi.org/10.1038/s41598-024-67826-9
- Figueroa et al. (2010). DNA methylation signatures identify biologically distinct subtypes in AML. *Cancer Cell*, 17(1), 13–27.
- TCGA Research Network (2013). Genomic and epigenomic landscapes of adult de novo AML. *NEJM*, 368(22), 2059–2074.
