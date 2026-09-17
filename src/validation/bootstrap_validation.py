"""
Master Prompt Section 8, Level 3: Harrell bootstrap optimism correction
(patient-level resampling with replacement, same n per resample).

For each model:
  1. Fit on the ORIGINAL cohort -> apparent C-index (C_app).
  2. For b in 1..B: resample patients with replacement (n same as original);
     fit the model on the bootstrap sample; evaluate that bootstrap-fitted
     model BOTH on the bootstrap sample itself (C_boot) and on the
     ORIGINAL cohort (C_orig). optimism_b = C_boot - C_orig.
  3. Bootstrap-corrected estimate = C_app - mean(optimism_b).
  4. 95% CI reported from the percentile distribution of (C_app -
     optimism_b) across the B resamples -- NOT the naive in-sample C_app
     alone, and NOT hidden: with only 11 events, this interval is wide by
     construction and is reported as such, per explicit instruction.

Resample counts (documented, not silently reduced):
  - Family-only Weibull, penalized Cox, XGBoost AFT: full B=500 (each
    refit costs well under a second; verified below).
  - Primary hierarchical Bayesian Weibull AFT (see bootstrap_bayesian.py):
    an earlier version of this bootstrap used MAP (maximum a posteriori)
    point estimates per resample instead of full MCMC, which meant the
    primary model's row was not evaluated the same way as the other four
    (a real methodological asymmetry, not just a resample-count
    difference). Superseded: bootstrap_bayesian.py now uses the SAME full
    4-chain, 2000-tune+2000-draw MCMC procedure as the apparent fit for
    every resample -- methodologically identical to how the other four
    comparators are bootstrapped. The only remaining difference is B=100
    (not 500): full-MCMC-per-resample is calibrated at ~73s/resample on
    this environment (PyTensor's C backend unavailable -- no MSVC Build
    Tools -- forcing the pure-Python fallback), so B=500 would take
    ~10.1 hours; B=100 (~2 hours) was the chosen tradeoff to keep the
    FITTING METHOD identical while keeping wall-clock time reasonable.
    The resulting interval is consequently noisier than the B=500
    comparators' intervals -- stated explicitly in
    reports/head_a_validation_report.md, not hidden.
  - Literature-prior-only (null): NOT bootstrapped. This model has zero
    fitted parameters -- its prediction is the literature point regardless
    of which patients are resampled, so there is no overfitting for a
    bootstrap to detect optimism about. Its single C-index (computed once,
    on the full original cohort) is reported as-is, with this reasoning
    stated explicitly rather than a fabricated confidence interval.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from lifelines import CoxPHFitter, WeibullFitter
from lifelines.utils import concordance_index

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "durability"))
from fit_literature_priors import SOURCES  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 20260917


def load_cohort():
    features = pd.read_csv("data_processed_patient_features.csv")
    labels = pd.read_csv("data_processed_patient_labels.csv")
    d = features.merge(labels, on="profile_key")
    d = d[d["index_approach"].notna()].reset_index(drop=True)
    d["t_approx"] = np.where(d["event"] == 1, (d["t_lower"] + d["t_upper"]) / 2, d["t_lower"])
    d["t_approx"] = d["t_approx"].clip(lower=0.5)
    return d


def resample(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.integers(0, len(d), len(d))
    out = d.iloc[idx].reset_index(drop=True)
    out["profile_key"] = [f"boot_{i}" for i in range(len(out))]
    return out


def percentile_ci(app: float, optimisms: list[float]) -> tuple[float, float, float]:
    corrected = app - float(np.mean(optimisms))
    dist = app - np.array(optimisms)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return corrected, float(lo), float(hi)


# --------------------------------------------------------------------------
# Family-only Weibull
# --------------------------------------------------------------------------
def _family_only_risk_scores(train: pd.DataFrame, eval_d: pd.DataFrame) -> np.ndarray | None:
    scores = np.full(len(eval_d), np.nan)
    for fam, sub in train.groupby("valve_family", dropna=True):
        if sub["event"].sum() == 0:
            continue  # undefined for this family in this resample -- those rows stay NaN
        wf = WeibullFitter()
        try:
            wf.fit(sub["t_approx"], event_observed=sub["event"])
        except Exception:
            continue
        mask = (eval_d["valve_family"] == fam).to_numpy()
        median_surv = wf.lambda_ * (np.log(2)) ** (1 / wf.rho_)
        scores[mask] = median_surv
    return scores


def bootstrap_family_only(d: pd.DataFrame, B: int = 500) -> dict:
    known = d[d["valve_family_known"]].reset_index(drop=True)
    app_scores = _family_only_risk_scores(known, known)
    valid = ~np.isnan(app_scores)
    c_app = concordance_index(known["t_approx"][valid], app_scores[valid], known["event"][valid])

    rng = np.random.default_rng(SEED)
    optimisms, n_undefined = [], 0
    for _ in range(B):
        boot = resample(known, rng)
        if boot["event"].sum() == 0:
            n_undefined += 1
            continue
        boot_scores = _family_only_risk_scores(boot, boot)
        orig_scores = _family_only_risk_scores(boot, known)
        v_boot = ~np.isnan(boot_scores)
        v_orig = ~np.isnan(orig_scores)
        if v_boot.sum() < 2 or v_orig.sum() < 2 or boot["event"][v_boot].sum() == 0 or known["event"][v_orig].sum() == 0:
            n_undefined += 1
            continue
        try:
            c_boot = concordance_index(boot["t_approx"][v_boot], boot_scores[v_boot], boot["event"][v_boot])
            c_orig = concordance_index(known["t_approx"][v_orig], orig_scores[v_orig], known["event"][v_orig])
        except Exception:
            n_undefined += 1
            continue
        optimisms.append(c_boot - c_orig)

    corrected, lo, hi = percentile_ci(c_app, optimisms)
    return dict(model="valve_family_only_weibull", apparent_c_index=float(c_app),
                bootstrap_corrected_c_index=corrected, ci_95_low=lo, ci_95_high=hi,
                n_resamples_used=len(optimisms), n_resamples_undefined=n_undefined, B=B)


# --------------------------------------------------------------------------
# Penalized Cox
# --------------------------------------------------------------------------
def _cox_covariates(sub: pd.DataFrame) -> pd.DataFrame:
    cov = sub[["t_approx", "event"]].copy()
    cov["ppm_proxy_flag"] = sub["ppm_proxy_flag"].fillna(False).astype(int)
    for a in ["TAVR", "ViV-TAVR"]:
        cov[f"approach_{a}"] = (sub["index_approach"] == a).astype(int)
    return cov


def bootstrap_cox(d: pd.DataFrame, B: int = 500) -> dict:
    cph_app = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    cph_app.fit(_cox_covariates(d), duration_col="t_approx", event_col="event")
    c_app = concordance_index(d["t_approx"], -cph_app.predict_partial_hazard(_cox_covariates(d)), d["event"])

    rng = np.random.default_rng(SEED)
    optimisms, n_failed = [], 0
    for _ in range(B):
        boot = resample(d, rng)
        try:
            cph_b = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
            cph_b.fit(_cox_covariates(boot), duration_col="t_approx", event_col="event")
            c_boot = concordance_index(boot["t_approx"], -cph_b.predict_partial_hazard(_cox_covariates(boot)), boot["event"])
            c_orig = concordance_index(d["t_approx"], -cph_b.predict_partial_hazard(_cox_covariates(d)), d["event"])
        except Exception:
            n_failed += 1
            continue
        optimisms.append(c_boot - c_orig)

    corrected, lo, hi = percentile_ci(c_app, optimisms)
    return dict(model="penalized_cox_elastic_net", apparent_c_index=float(c_app),
                bootstrap_corrected_c_index=corrected, ci_95_low=lo, ci_95_high=hi,
                n_resamples_used=len(optimisms), n_resamples_failed=n_failed, B=B)


# --------------------------------------------------------------------------
# XGBoost AFT
# --------------------------------------------------------------------------
XGB_PARAMS = dict(objective="survival:aft", eval_metric="aft-nloglik",
                   aft_loss_distribution="normal", aft_loss_distribution_scale=1.2,
                   tree_method="hist", max_depth=2, eta=0.05, seed=SEED)


def _fit_xgb_aft(sub: pd.DataFrame):
    feat_cols = ["ppm_proxy_flag"]
    sub = sub.copy()
    sub["ppm_proxy_flag"] = sub["ppm_proxy_flag"].fillna(False).astype(float)
    for a in ["TAVR", "ViV-TAVR"]:
        sub[f"approach_{a}"] = (sub["index_approach"] == a).astype(float)
        feat_cols.append(f"approach_{a}")
    dtrain = xgb.DMatrix(sub[feat_cols].to_numpy())
    dtrain.set_float_info("label_lower_bound", sub["t_lower"].to_numpy(dtype=float))
    dtrain.set_float_info("label_upper_bound", sub["t_upper"].replace(float("inf"), np.inf).to_numpy(dtype=float))
    booster = xgb.train(XGB_PARAMS, dtrain, num_boost_round=200)
    return booster, feat_cols


def _xgb_predict(booster, feat_cols, sub: pd.DataFrame) -> np.ndarray:
    sub = sub.copy()
    sub["ppm_proxy_flag"] = sub["ppm_proxy_flag"].fillna(False).astype(float)
    for a in ["TAVR", "ViV-TAVR"]:
        sub[f"approach_{a}"] = (sub["index_approach"] == a).astype(float)
    return booster.predict(xgb.DMatrix(sub[feat_cols].to_numpy()))


def bootstrap_xgb_aft(d: pd.DataFrame, B: int = 500) -> dict:
    booster_app, feat_cols = _fit_xgb_aft(d)
    pred_app = _xgb_predict(booster_app, feat_cols, d)
    c_app = concordance_index(d["t_approx"], pred_app, d["event"])

    rng = np.random.default_rng(SEED)
    optimisms, n_failed = [], 0
    for _ in range(B):
        boot = resample(d, rng)
        try:
            booster_b, fc = _fit_xgb_aft(boot)
            c_boot = concordance_index(boot["t_approx"], _xgb_predict(booster_b, fc, boot), boot["event"])
            c_orig = concordance_index(d["t_approx"], _xgb_predict(booster_b, fc, d), d["event"])
        except Exception:
            n_failed += 1
            continue
        optimisms.append(c_boot - c_orig)

    corrected, lo, hi = percentile_ci(c_app, optimisms)
    return dict(model="xgboost_aft", apparent_c_index=float(c_app),
                bootstrap_corrected_c_index=corrected, ci_95_low=lo, ci_95_high=hi,
                n_resamples_used=len(optimisms), n_resamples_failed=n_failed, B=B)


# --------------------------------------------------------------------------
# Null / literature-prior-only
# --------------------------------------------------------------------------
def null_prior_only(d: pd.DataFrame) -> dict:
    approach_prior_map = {"SAVR": "SAVR", "TAVR": "TAVR", "ViV-TAVR": "TAVR", "ViV-SAVR": "SAVR"}
    with open("config/priors.yaml", encoding="utf-8") as f:
        priors = yaml.safe_load(f)
    scale = d["index_approach"].map(lambda a: priors["approach_level"][approach_prior_map.get(a, "SAVR")]["scale_lambda_years"])
    shape = d["index_approach"].map(lambda a: priors["approach_level"][approach_prior_map.get(a, "SAVR")]["shape_k"])
    median_surv = scale * (np.log(2)) ** (1 / shape)
    c = concordance_index(d["t_approx"], median_surv, d["event"])
    return dict(model="literature_prior_only_null", apparent_c_index=float(c),
                bootstrap_corrected_c_index=None, ci_95_low=None, ci_95_high=None,
                note="Not bootstrapped -- zero fitted parameters, no overfitting for a bootstrap to detect; "
                     "see module docstring.")


if __name__ == "__main__":
    d = load_cohort()
    results = []

    print("=== Null / literature-prior-only ===")
    r = null_prior_only(d)
    print(r); results.append(r)

    print("\n=== Valve-family-only Weibull (B=500) ===")
    t0 = time.time()
    r = bootstrap_family_only(d, B=500)
    print(r, f"({time.time()-t0:.0f}s)"); results.append(r)

    print("\n=== Penalized Cox elastic net (B=500) ===")
    t0 = time.time()
    r = bootstrap_cox(d, B=500)
    print(r, f"({time.time()-t0:.0f}s)"); results.append(r)

    print("\n=== XGBoost AFT (B=500) ===")
    t0 = time.time()
    r = bootstrap_xgb_aft(d, B=500)
    print(r, f"({time.time()-t0:.0f}s)"); results.append(r)

    import json
    results = json.loads(json.dumps(results, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    with open("reports/head_a_bootstrap_frequentist.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False, allow_unicode=True)
    print("\nwrote reports/head_a_bootstrap_frequentist.yaml")
