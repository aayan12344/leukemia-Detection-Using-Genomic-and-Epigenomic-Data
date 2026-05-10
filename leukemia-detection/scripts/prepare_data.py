"""
prepare_data.py
===============
Parse any number of GEO series matrix files and combine them into
clean CSVs ready for the training pipeline.

Usage
-----
python scripts/prepare_data.py \
    --datasets data/raw/GSE58477_series_matrix.txt.gz \
               data/raw/GSE63409_series_matrix.txt.gz
"""

import gzip
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO


LABEL_RULES = {
    "GSE58477": lambda src, char: (
        0 if ("normal" in src.lower() or "cd34" in src.lower())
        else 1 if ("leukemic" in src.lower() or "blast" in src.lower())
        else None
    ),
    "GSE63409": lambda src, char: (
        0 if ("normal" in char.lower() or "normal bone marrow" in src.lower())
        else 1 if ("aml" in char.lower() or "aml patient" in src.lower())
        else None
    ),
}

def _default_label_rule(src, char):
    s = src.lower() + " " + char.lower()
    if any(w in s for w in ["normal", "healthy", "control", "cd34", "bone marrow donor"]):
        return 0
    if any(w in s for w in ["leukemia", "leukemic", "blast", "aml", "all", "cml", "cll"]):
        return 1
    return None

def _get_label_rule(filepath):
    name = Path(filepath).name
    for accession, rule in LABEL_RULES.items():
        if accession in name:
            return rule
    return _default_label_rule

def parse_geo_matrix(filepath, top_cpgs=10000):
    dataset_name = Path(filepath).name.split("_")[0]
    label_rule   = _get_label_rule(filepath)

    sample_ids   = []
    source_names = []
    char_buffer  = {}
    data_lines   = []

    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
        in_table = False
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_source_name_ch1"):
                source_names = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
                for i, v in enumerate(vals):
                    char_buffer.setdefault(i, []).append(v)
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                data_lines.append(line)

    if not sample_ids:
        raise ValueError(f"No sample IDs found in {filepath}.")

    characteristics = [" | ".join(char_buffer.get(i, [])) for i in range(len(sample_ids))]
    labels_raw = {sid: label_rule(src, char)
                  for sid, src, char in zip(sample_ids, source_names, characteristics)}

    if not any(v in (0, 1) for v in labels_raw.values()):
        raise ValueError(f"No samples labelled in {dataset_name}. Add a rule to LABEL_RULES.")

    keep_samples = {sid for sid, lbl in labels_raw.items() if lbl is not None}
    labels       = {sid: lbl for sid, lbl in labels_raw.items() if lbl is not None}

    header_line = data_lines[0]
    rows        = data_lines[1:]
    row_var     = {}
    BATCH       = 10000

    for i in range(0, len(rows), BATCH):
        df_b = pd.read_csv(StringIO(header_line + "".join(rows[i:i+BATCH])),
                           sep="\t", index_col=0)
        df_b.index   = [x.strip('"') for x in df_b.index]
        df_b.columns = [x.strip('"') for x in df_b.columns]
        df_b = df_b[[c for c in df_b.columns if c in keep_samples]]
        for cpg, row in df_b.iterrows():
            row_var[cpg] = np.nanvar(row.values.astype(float))

    top_set  = set(sorted(row_var, key=row_var.get, reverse=True)[:min(top_cpgs, len(row_var))])
    selected = [header_line] + [l for l in rows if l.split("\t")[0].strip('"') in top_set]

    df = pd.read_csv(StringIO("".join(selected)), sep="\t", index_col=0)
    df.index   = [x.strip('"') for x in df.index]
    df.columns = [x.strip('"') for x in df.columns]
    df = df[[c for c in df.columns if c in keep_samples]]

    return df.T, labels, dataset_name








def combine_datasets(parsed, final_cpgs=5000):
    common_cpgs = parsed[0][0].columns
    for meth_df, _, _ in parsed[1:]:
        common_cpgs = common_cpgs.intersection(meth_df.columns)

    if len(common_cpgs) == 0:
        raise ValueError("No common CpGs found. Make sure all datasets use the same array platform.")

    meth_parts, label_parts = [], []
    for meth_df, labels, dataset_name in parsed:
        aligned = meth_df[common_cpgs]
        meth_parts.append(aligned)
        ldf = pd.DataFrame({"label": labels, "dataset": dataset_name}, index=list(labels.keys()))
        label_parts.append(ldf.loc[aligned.index])

    combined_meth   = pd.concat(meth_parts,  axis=0)
    combined_labels = pd.concat(label_parts, axis=0)
    top_final       = combined_meth.var(axis=0, skipna=True).nlargest(
                          min(final_cpgs, len(common_cpgs))).index
    return combined_meth[top_final], combined_labels






def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",   nargs="+", required=True)
    p.add_argument("--output",     default="data/processed/")
    p.add_argument("--top_cpgs",   type=int, default=10000)
    p.add_argument("--final_cpgs", type=int, default=5000)
    return p.parse_args()





def main():
    args = parse_args()

    for path in args.datasets:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    parsed = [parse_geo_matrix(p, top_cpgs=args.top_cpgs) for p in args.datasets]
    combined_meth, combined_labels = combine_datasets(parsed, final_cpgs=args.final_cpgs)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_meth.to_csv(out_dir / "methylation_combined.csv")
    combined_labels.to_csv(out_dir / "labels_combined.csv")

if __name__ == "__main__":
    main()