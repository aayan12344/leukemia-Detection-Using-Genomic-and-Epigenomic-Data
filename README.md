# Leukemia Detection via DNA Methylation

Binary classification of AML leukemia vs. normal bone marrow

**Dataset:** 146 samples (106 AML · 40 Normal) across two GEO cohorts — GSE58477 + GSE63409  
**Best result:** AUC = 1.0 · F1 = 0.93 · Accuracy = 90%

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/leukemia-detection.git
cd leukemia-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

```bash
python scripts/train.py --mode csv \
  --methylation data/processed/methylation_combined.csv \
  --labels      data/processed/labels_combined.csv
```

Results and figures are saved to `results/`.

To run on simulated data (no files needed):
```bash
python scripts/train.py --mode simulate
```

---

## Project Structure

```
├── src/
│   ├── data_loader.py        # Load GEO / simulate data
│   ├── preprocessing.py      # Imputation and scaling
│   ├── feature_selection.py  # ANOVA, variance, LASSO
│   ├── models.py             # Classifiers and CV training
│   ├── evaluation.py         # Metrics and reporting
│   └── visualization.py      # All plots
├── scripts/
│   └── train.py              # Main entry point
├── data/
│   └── processed/            # Not committed — see .gitignore
└── results/
    ├── figures/              # Generated plots
    └── models/               # Saved model files
```

---

## Pipeline

```
Raw GEO Data
    ↓
Feature Filtering (remove low-variance / high-missing CpGs)
    ↓
Stratified Train / Test Split  ← happens before any normalization
    ↓
Imputation + Z-score Scaling   ← fit on train, applied to both
    ↓
ANOVA Feature Selection        ← top 500 CpGs, train only
    ↓
5-Fold Cross-Validation Training (SVM, RF, GBM, LR, MLP)
    ↓
Final Evaluation on Test Set
```

---

## Data

Data is downloaded from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) and not committed to this repo. To reproduce:

1. Download series matrix files from GSE58477 and GSE63409
2. Place in `data/raw/`
3. Run the parsing script or follow the steps in `src/data_loader.py`

---

## References

- Hovestadt et al. (2025). Rapid epigenomic classification of acute leukemia. *Nature Genetics.* [doi:10.1038/s41588-025-02321-z](https://doi.org/10.1038/s41588-025-02321-z)
- Gandomi et al. (2024). Attention-based deep learning for ALL classification. *Scientific Reports.* [doi:10.1038/s41598-024-67826-9](https://doi.org/10.1038/s41598-024-67826-9)