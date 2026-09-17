"""
Section 7 explainability outputs for Head A.

1. Forest plot of posterior time ratios: each approach and valve-family
   scale parameter, expressed as a ratio to the SAVR population reference
   (exp(mu_group - mu_SAVR)), with 94% highest-density intervals. A time
   ratio > 1 means "predicted to last longer than the SAVR reference"; < 1
   means shorter.
2. SHAP summary for the XGBoost AFT comparator (Section 5.4 lists this
   comparator, not the primary model, as the one with a scalable
   SHAP/TreeExplainer path). Reported with an explicit honesty caveat: this
   comparator's own C-index (0.519, see reports/head_a_comparator_cindex.yaml)
   is barely above chance at this sample size, so its SHAP ranking should be
   read as "what the model leaned on to get a near-null result," not as
   validated risk-factor importance.
3. Mandatory disclaimer (Section 7): every output states factors are
   associations from n=99/11-events, not causal effects.
"""
from __future__ import annotations

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

DISCLAIMER = ("Associations only, from n=99 patients / 11 events (Head A cohort) — not causal effects. "
              "See reports/head_a_validation_report.md for full uncertainty and limitations.")


def forest_plot_time_ratios(nc_path="reports/head_a_posterior.nc", out_path="reports/head_a_forest_plot.png"):
    idata = az.from_netcdf(nc_path)
    post = idata.posterior
    mu_approach = post["mu_approach"]
    savr_ref = mu_approach.sel(approach="SAVR")
    approach_names = mu_approach.coords["approach"].values.tolist()

    family_offset = post["family_offset"]
    family_names = family_offset.coords["family"].values.tolist()
    # family time ratio relative to ITS OWN approach mean (the offset is
    # already defined that way in the model), not relative to SAVR overall.
    family_ratio = np.exp(family_offset)

    labels, medians, los, his = [], [], [], []
    for a in approach_names:
        ratio = np.exp(mu_approach.sel(approach=a) - savr_ref)
        flat = ratio.values.flatten()
        labels.append(f"approach: {a} (vs SAVR reference)")
        medians.append(np.median(flat))
        lo, hi = np.percentile(flat, [5.5, 94.5])
        los.append(lo); his.append(hi)
    for fam in family_names:
        flat = family_ratio.sel(family=fam).values.flatten()
        labels.append(f"family offset: {fam.replace('_', ' ')}")
        medians.append(np.median(flat))
        lo, hi = np.percentile(flat, [5.5, 94.5])
        los.append(lo); his.append(hi)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 1.5))
    ax.errorbar(medians, y, xerr=[np.array(medians) - np.array(los), np.array(his) - np.array(medians)],
                fmt="o", color="#2b6cb0", ecolor="#63b3ed", capsize=4, markersize=6)
    ax.axvline(1.0, color="#718096", linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time ratio (posterior median, 89% credible interval)\n>1 = longer predicted durability than reference")
    ax.set_title("Head A — posterior time ratios, hierarchical Bayesian Weibull AFT")
    fig.text(0.02, 0.01, DISCLAIMER, fontsize=7, color="#4a5568")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def shap_summary_xgb_aft(model_path="reports/head_a_comparator_xgb_aft.json",
                          out_path="reports/head_a_shap_summary.png"):
    features = pd.read_csv("data_processed_patient_features.csv")
    labels = pd.read_csv("data_processed_patient_labels.csv")
    d = features.merge(labels, on="profile_key")
    d = d[d["index_approach"].notna()].reset_index(drop=True)
    feat_cols = ["ppm_proxy_flag"]
    d["ppm_proxy_flag"] = d["ppm_proxy_flag"].fillna(False).astype(float)
    for a in ["TAVR", "ViV-TAVR"]:
        d[f"approach_{a}"] = (d["index_approach"] == a).astype(float)
        feat_cols.append(f"approach_{a}")
    X = d[feat_cols]

    booster = xgb.Booster()
    booster.load_model(model_path)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    fig = plt.figure(figsize=(7, 4))
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.title("XGBoost AFT comparator — SHAP feature importance\n"
              "(comparator C-index = 0.519, near chance at this n — read as diagnostic, not validated ranking)",
              fontsize=9)
    plt.gcf().text(0.02, 0.01, DISCLAIMER, fontsize=7, color="#4a5568")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return dict(zip(feat_cols, np.abs(shap_values).mean(axis=0).tolist()))


if __name__ == "__main__":
    forest_plot_time_ratios()
    mean_abs_shap = shap_summary_xgb_aft()
    print("Mean |SHAP| per feature (XGBoost AFT comparator):", mean_abs_shap)
