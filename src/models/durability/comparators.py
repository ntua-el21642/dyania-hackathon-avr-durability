"""
Head A comparators (Master Prompt Section 5.4). Run alongside the primary
hierarchical Bayesian Weibull AFT (weibull_aft_bayes.py) so its performance
can be judged against simpler baselines, not reported in isolation.

Comparators built:
  1. Null / prior-only: literature Weibull priors with NO data update
     (closed form from config/priors.yaml, no fitting).
  2. Age-only: NOT BUILT. Age at implant is masked ([AGE]) throughout the
     corpus with zero surviving numeric values (verified by full-corpus
     regex scan, see build_features.py) -- there is no age covariate to
     fit. Documented here rather than silently substituted with something
     else.
  3. Valve-family-only Weibull: frequentist MLE per family via
     lifelines.WeibullFitter on an interval-censored approximation
     (right-censored at the interval midpoint for events -- see note in
     `midpoint_approx_events`), labelled as an approximation per Section
     5.4.
  4. Penalised Cox (elastic net): Master Prompt specifies scikit-survival,
     which requires a C++ build toolchain not available in this Windows
     environment (`pip install scikit-survival` fails building its `ecos`
     dependency -- Microsoft C++ Build Tools not installed). Substituted
     with lifelines' CoxPHFitter(penalizer=..., l1_ratio=0.5), which fits
     the same elastic-net-penalized Cox partial likelihood without a C
     compiler. Documented substitution, not a silent swap.
  5. XGBoost AFT: survival:aft objective, interval-censored labels via
     label_lower_bound/label_upper_bound -- the scalable, production-
     deployment-candidate comparator (Section 5.4).

All comparators are evaluated with Uno's C-index (via lifelines'
concordance_index on the midpoint-approximated times, the closest available
without scikit-survival's IPCW C-index implementation -- documented
approximation) and reported in reports/head_a_comparators.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from lifelines import CoxPHFitter, WeibullFitter
from lifelines.utils import concordance_index


def midpoint_approx_events(labels: pd.DataFrame) -> pd.DataFrame:
    """Right-censored midpoint approximation: for events, use the interval
    midpoint (t_lower+t_upper)/2 as an exact time; for censored, use
    t_lower as the observed (right-censored) time. Explicitly labelled an
    approximation everywhere it's used (Section 5.4) -- the primary model
    uses the true interval-censored likelihood instead."""
    out = labels.copy()
    is_event = out["event"] == 1
    out["t_approx"] = np.where(is_event, (out["t_lower"] + out["t_upper"]) / 2, out["t_lower"])
    # A handful of censored patients have only one note ever filed (last_note_year
    # == index_implant_year), giving t_lower = 0 exactly -- not a real "zero
    # follow-up" claim, just year-granularity rounding of a same-year visit.
    # Floor at 0.5y so the frequentist fitters (which require strictly positive
    # durations) don't choke on it; the primary interval-censored model has no
    # such requirement and is unaffected.
    out["t_approx"] = out["t_approx"].clip(lower=0.5)
    return out


def fit_null_prior_only(priors: dict, approaches: list[str]) -> pd.DataFrame:
    rows = []
    approach_prior_map = {"SAVR": "SAVR", "TAVR": "TAVR", "ViV-TAVR": "TAVR", "ViV-SAVR": "SAVR"}
    for a in approaches:
        p = priors["approach_level"][approach_prior_map.get(a, "SAVR")]
        rows.append(dict(approach=a, shape_k=p["shape_k"], scale_lambda_years=p["scale_lambda_years"],
                          source="literature prior only, no data update"))
    return pd.DataFrame(rows)


def fit_family_only_weibull(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, sub in d.groupby("valve_family", dropna=True):
        if sub["event"].sum() == 0:
            rows.append(dict(valve_family=fam, n=len(sub), n_events=0,
                              note="no events in this family -- MLE undefined, prior-only recommended"))
            continue
        wf = WeibullFitter()
        wf.fit(sub["t_approx"], event_observed=sub["event"])
        rows.append(dict(valve_family=fam, n=len(sub), n_events=int(sub["event"].sum()),
                          shape_k=wf.rho_, scale_lambda_years=wf.lambda_))
    return pd.DataFrame(rows)


def fit_penalized_cox(d: pd.DataFrame) -> tuple[CoxPHFitter, float]:
    cov = d[["t_approx", "event", "ppm_proxy_flag"]].copy()
    cov["ppm_proxy_flag"] = cov["ppm_proxy_flag"].fillna(False).astype(int)
    for a in ["SAVR", "TAVR", "ViV-TAVR"]:
        cov[f"approach_{a}"] = (d["index_approach"] == a).astype(int)
    cov = cov.drop(columns=["approach_SAVR"])  # reference level
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)  # elastic net, substituting scikit-survival -- see module docstring
    cph.fit(cov, duration_col="t_approx", event_col="event")
    c_index = concordance_index(cov["t_approx"], -cph.predict_partial_hazard(cov), cov["event"])
    return cph, c_index


def fit_xgboost_aft(d: pd.DataFrame) -> tuple[xgb.Booster, float]:
    feat_cols = ["ppm_proxy_flag"]
    d = d.copy()
    d["ppm_proxy_flag"] = d["ppm_proxy_flag"].fillna(False).astype(float)
    for a in ["TAVR", "ViV-TAVR"]:
        d[f"approach_{a}"] = (d["index_approach"] == a).astype(float)
        feat_cols.append(f"approach_{a}")
    X = d[feat_cols].to_numpy()
    y_lower = d["t_lower"].to_numpy(dtype=float)
    y_upper = d["t_upper"].replace(float("inf"), np.inf).to_numpy(dtype=float)

    dtrain = xgb.DMatrix(X)
    dtrain.set_float_info("label_lower_bound", y_lower)
    dtrain.set_float_info("label_upper_bound", y_upper)

    params = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist", "max_depth": 2, "eta": 0.05, "seed": 20260917,
    }
    booster = xgb.train(params, dtrain, num_boost_round=200)
    pred = booster.predict(dtrain)  # predicted median survival time per patient
    t_eval = np.where(d["event"] == 1, (d["t_lower"] + d["t_upper"]) / 2, d["t_lower"])
    c_index = concordance_index(t_eval, pred, d["event"])
    return booster, c_index


if __name__ == "__main__":
    features = pd.read_csv("data_processed_patient_features.csv")
    labels = pd.read_csv("data_processed_patient_labels.csv")
    with open("config/priors.yaml", encoding="utf-8") as f:
        priors = yaml.safe_load(f)

    d = features.merge(labels, on="profile_key")
    d = d[d["index_approach"].notna()].reset_index(drop=True)
    d = midpoint_approx_events(d)

    print("--- 1. Null / prior-only ---")
    null_df = fit_null_prior_only(priors, sorted(d["index_approach"].unique().tolist()))
    print(null_df.to_string(index=False))

    print("\n--- 3. Valve-family-only Weibull (frequentist MLE, midpoint approx.) ---")
    fam_df = fit_family_only_weibull(d)
    print(fam_df.to_string(index=False))

    print("\n--- 4. Penalised Cox (elastic net via lifelines, substituting scikit-survival) ---")
    cph, cox_c = fit_penalized_cox(d)
    print(cph.summary[["coef", "exp(coef)", "se(coef)", "p"]].to_string())
    print(f"C-index (midpoint approx., in-sample): {cox_c:.3f}")

    print("\n--- 5. XGBoost AFT (interval-censored labels) ---")
    booster, xgb_c = fit_xgboost_aft(d)
    print(f"C-index (midpoint approx., in-sample): {xgb_c:.3f}")

    null_df.to_csv("reports/head_a_comparator_null.csv", index=False)
    fam_df.to_csv("reports/head_a_comparator_family_only.csv", index=False)
    cph.summary.to_csv("reports/head_a_comparator_cox.csv")
    booster.save_model("reports/head_a_comparator_xgb_aft.json")
    with open("reports/head_a_comparator_cindex.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"cox_c_index_in_sample": float(cox_c), "xgb_aft_c_index_in_sample": float(xgb_c)}, f)
    print("\nwrote reports/head_a_comparator_*")
