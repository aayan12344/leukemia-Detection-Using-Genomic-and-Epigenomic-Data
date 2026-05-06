"""
leukemia-detection
==================
Binary classification of Leukemia vs. Normal using DNA methylation
and gene expression multi-omic data.

Modules
-------
data_loader        : Load / simulate TCGA & GEO datasets
preprocessing      : Imputation, scaling, sample/feature filtering
feature_selection  : ANOVA, variance threshold, multi-omic selectors
models             : Classifier definitions and cross-validation training
evaluation         : Metrics computation and reporting
visualization      : All plotting utilities
"""

__version__ = "0.1.0"
__author__  = "Your Name"
