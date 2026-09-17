"""
Master Prompt Section 8, Level 1: NLP vs. physician adjudication metrics,
computed against the FULL 117-patient blind adjudication
(review/adjudication_form.xlsx), not just the 19-patient seed table.

Computes what this form can actually support:
  - Reintervention (BVF Stage 2) detector: precision, recall, F1.
  - HVD stage agreement: exact agreement rate + quadratic-weighted kappa
    (ordinal 0-3).
  - SVD-present (Y/N) agreement: Cohen's kappa.
  - Confidence-tier / BVF-stage disagreement catalogue, one row per
    disagreeing patient, with the pipeline's own rationale/evidence
    attached, for physician re-review -- NOT auto-resolved.

NOT computed, and not fabricated: "value accuracy" for extracted
gradients/size/LVEF and "temporal-attribution accuracy" (Master Prompt
Section 8 Level 1's full spec) -- the adjudication form as built captures
stage/phenotype/confidence determinations, not a field-by-field check of
individual extracted numeric values or their assigned years. That would
need a different review instrument (e.g. a value-level spot-check sheet)
that was never built. Reported as a gap, not estimated.
"""
from __future__ import annotations

import numpy as np
import openpyxl
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support


def load_form(path: str = "review/adjudication_form.xlsx") -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)
    ws = wb["Untitled"] if "Untitled" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(min_row=2, max_row=118, values_only=True))
    df = pd.DataFrame(rows, columns=[
        "profile_key", "n_notes", "notes_types_years", "svd_present_form",
        "hvd_stage_form", "phenotype_form", "bvf_stage_form", "confidence_form",
        "exclusion_form", "_blank", "reviewer_notes",
    ])
    df = df.drop(columns=["_blank"])
    df["hvd_stage_form"] = pd.to_numeric(df["hvd_stage_form"], errors="coerce")
    df["bvf_stage_form"] = pd.to_numeric(df["bvf_stage_form"], errors="coerce")
    df["svd_present_form"] = df["svd_present_form"].astype(str).str.strip().str.upper()
    return df


def load_pipeline(path: str = "labels.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["svd_present_pipe"] = df["confidence_tier"].isin(["definite", "probable", "possible"])
    return df


def build_merged() -> pd.DataFrame:
    form = load_form()
    pipe = load_pipeline()
    m = form.merge(pipe, on="profile_key", how="inner")
    assert len(m) == 117, f"expected 117 matched patients, got {len(m)}"
    return m


def reintervention_metrics(m: pd.DataFrame) -> dict:
    y_true = (m["bvf_stage_form"] == 2).astype(int)
    y_pred = (m["bvf_stage"] == 2).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return dict(n=len(m), precision=float(p), recall=float(r), f1=float(f1), tp=tp, fp=fp, fn=fn, tn=tn)


def hvd_stage_agreement(m: pd.DataFrame) -> dict:
    both = m.dropna(subset=["hvd_stage_form", "hvd_stage"])
    exact = float((both["hvd_stage_form"] == both["hvd_stage"]).mean())
    kappa_w = cohen_kappa_score(both["hvd_stage_form"].astype(int), both["hvd_stage"].astype(int), weights="quadratic")
    return dict(n=len(both), exact_agreement=exact, weighted_kappa=float(kappa_w))


def svd_present_agreement(m: pd.DataFrame) -> dict:
    both = m.dropna(subset=["svd_present_pipe"])
    both = both[both["svd_present_form"].isin(["Y", "N"])]
    y_form = (both["svd_present_form"] == "Y").astype(int)
    y_pipe = both["svd_present_pipe"].astype(int)
    exact = float((y_form == y_pipe).mean())
    kappa = cohen_kappa_score(y_form, y_pipe)
    return dict(n=len(both), exact_agreement=exact, kappa=float(kappa))


def bvf_disagreements(m: pd.DataFrame) -> pd.DataFrame:
    dis = m[m["bvf_stage_form"] != m["bvf_stage"]].copy()
    cols = ["profile_key", "bvf_stage_form", "bvf_stage", "hvd_stage_form", "hvd_stage",
            "svd_present_form", "confidence_form", "confidence_tier",
            "reviewer_notes", "redo_evidence", "svd_explicit_evidence", "exclusion_reason", "rationale"]
    return dis[cols].reset_index(drop=True)


if __name__ == "__main__":
    m = build_merged()
    print("=== Reintervention (BVF Stage 2) detector vs. full 117-patient adjudication ===")
    print(reintervention_metrics(m))
    print()
    print("=== HVD stage agreement ===")
    print(hvd_stage_agreement(m))
    print()
    print("=== SVD-present (Y/N) agreement ===")
    print(svd_present_agreement(m))
    print()
    print("=== BVF-stage disagreements (n=%d) ===" % (m["bvf_stage_form"] != m["bvf_stage"]).sum())
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 220)
    print(bvf_disagreements(m).to_string())
