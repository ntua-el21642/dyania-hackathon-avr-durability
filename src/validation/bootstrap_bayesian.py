"""
Harrell bootstrap optimism correction for the PRIMARY model (hierarchical
Bayesian Weibull AFT) using the SAME full 4-chain, 2000-tune + 2000-draw
MCMC procedure as the one reported "apparent" fit (reports/head_a_posterior.nc)
-- i.e. no MAP shortcut. This removes the MAP-vs-full-MCMC methodological
asymmetry the first version of this script had against the other four
comparators.

What is still different from the other four comparators, and why (stated
explicitly, not hidden): B=100 here, not B=500. Calibrated directly on this
environment (PYTENSOR_FLAGS="cxx=" pure-Python fallback, no C compiler
available): one full 4-chain 2000+2000 refit on a bootstrap-resampled
dataset measured at ~73s (vs. ~9.5s for a MAP refit). 500 resamples at that
rate would take ~10.1 hours; B=100 (~2 hours) was chosen as the resample
count that keeps the FITTING METHOD identical across all five models
(the thing that actually matters for a fair comparison) while keeping
wall-clock time reasonable for this session. The resulting interval is
therefore noisier (wider, less stable percentile estimates) than the
B=500 comparators' intervals -- stated in the output and in
reports/head_a_validation_report.md, not smoothed over. A full B=500
re-run remains possible as a background job if time allows later.

Run with: PYTENSOR_FLAGS="cxx=" PYTHONIOENCODING=utf-8 python src/validation/bootstrap_bayesian.py
Expected wall-clock: ~100 x ~73s =~ 2 hours on this environment.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import yaml
from lifelines.utils import concordance_index

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "durability"))
from weibull_aft_bayes import build_model  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20260917
B = 100
MCMC_KWARGS = dict(draws=2000, tune=2000, chains=4, cores=1, target_accept=0.95, progressbar=False)
CHECKPOINT_PATH = "reports/head_a_bootstrap_bayesian_checkpoint.csv"


def load_cohort():
    features = pd.read_csv("data_processed_patient_features.csv")
    labels = pd.read_csv("data_processed_patient_labels.csv")
    d = features.merge(labels, on="profile_key")
    d = d[d["index_approach"].notna()].reset_index(drop=True)
    d["t_approx"] = np.where(d["event"] == 1, (d["t_lower"] + d["t_upper"]) / 2, d["t_lower"])
    d["t_approx"] = d["t_approx"].clip(lower=0.5)
    return d


def resample(features: pd.DataFrame, labels: pd.DataFrame, rng: np.random.Generator):
    idx = rng.integers(0, len(features), len(features))
    bf = features.iloc[idx].reset_index(drop=True)
    bl = labels.iloc[idx].reset_index(drop=True)
    new_keys = [f"boot_{i}" for i in range(len(bf))]
    bf["profile_key"] = new_keys
    bl["profile_key"] = new_keys
    return bf, bl


def median_survival_from_posterior(idata, d_eval: pd.DataFrame,
                                    approaches: list[str], families: list[str]) -> np.ndarray:
    """Compute median survival for arbitrary evaluation rows from a fitted
    MCMC posterior's POSTERIOR MEAN (mu_approach, family_offset, beta_ppm,
    shape_k) -- pure numpy, so it can score the ORIGINAL cohort using a
    model that was fit (via full MCMC) on a bootstrap sample."""
    post = idata.posterior
    approach_idx = {a: i for i, a in enumerate(approaches)}
    family_idx = {f: i for i, f in enumerate(families)}
    k = float(post["shape_k"].mean())
    mu_approach = post["mu_approach"].mean(dim=["chain", "draw"]).values
    family_offset = post["family_offset"].mean(dim=["chain", "draw"]).values
    beta_ppm = float(post["beta_ppm"].mean())

    log_scale = np.array([mu_approach[approach_idx.get(a, 0)] for a in d_eval["index_approach"]])
    fam_term = np.array([
        family_offset[family_idx[f]] if (isinstance(f, str) and f in family_idx) else 0.0
        for f in d_eval["valve_family"]
    ])
    ppm = d_eval["ppm_proxy_flag"].fillna(False).to_numpy(dtype=float)
    log_scale = log_scale + fam_term + beta_ppm * ppm
    scale = np.exp(log_scale)
    return scale * (np.log(2)) ** (1.0 / k)


def run():
    d_full = load_cohort()
    features_full = pd.read_csv("data_processed_patient_features.csv")
    features_full = features_full[features_full["profile_key"].isin(d_full["profile_key"])].reset_index(drop=True)
    labels_full = pd.read_csv("data_processed_patient_labels.csv")
    labels_full = labels_full[labels_full["profile_key"].isin(d_full["profile_key"])].reset_index(drop=True)
    with open("config/priors.yaml", encoding="utf-8") as f:
        priors = yaml.safe_load(f)

    # --- apparent: use the ALREADY-FITTED full-MCMC posterior (the actually reported model) ---
    idata_app = az.from_netcdf("reports/head_a_posterior.nc")
    med_surv_app = idata_app.posterior["median_survival_years"].mean(dim=["chain", "draw"]).values
    c_app = concordance_index(d_full["t_approx"], med_surv_app, d_full["event"])
    print(f"Apparent C-index (full MCMC posterior, reported model): {c_app:.4f}")

    # --- resume from checkpoint if one exists (a prior run was killed by the
    # environment at 70/100 with no error -- see git history / conversation
    # for context; this makes that non-repeatable) ---
    ckpt_path = Path(CHECKPOINT_PATH)
    if ckpt_path.exists():
        ckpt_df = pd.read_csv(ckpt_path)
        start_b = len(ckpt_df)
        optimisms = ckpt_df["optimism"].tolist()
        max_rhats = ckpt_df["max_rhat"].dropna().tolist()
        n_divergences = ckpt_df["n_divergences"].dropna().tolist()
        print(f"Resuming from checkpoint: {start_b} resamples already done.")
    else:
        with open(ckpt_path, "w", encoding="utf-8") as f:
            f.write("resample,optimism,max_rhat,n_divergences\n")
        start_b, optimisms, max_rhats, n_divergences = 0, [], [], []

    rng = np.random.default_rng(SEED + start_b)  # different seed offset per resume batch -- fine for a bootstrap CI, doesn't need to replay the exact same draws
    t_start = time.time()
    for b in range(start_b, B):
        bf, bl = resample(features_full, labels_full, rng)
        try:
            model_b, meta_b = build_model(bf, bl, priors)
            with model_b:
                idata_b = pm.sample(random_seed=SEED + b, **MCMC_KWARGS)
        except Exception as e:
            print(f"  resample {b}: FAILED to fit ({e}); skipping")
            continue

        d_boot = bf.merge(bl, on="profile_key")
        d_boot = d_boot[d_boot["index_approach"].notna()].reset_index(drop=True)
        d_boot["t_approx"] = np.clip(
            np.where(d_boot["event"] == 1, (d_boot["t_lower"] + d_boot["t_upper"]) / 2, d_boot["t_lower"]), 0.5, None)

        pred_boot = median_survival_from_posterior(idata_b, d_boot, meta_b["approaches"], meta_b["families"])
        pred_orig = median_survival_from_posterior(idata_b, d_full, meta_b["approaches"], meta_b["families"])

        c_boot = concordance_index(d_boot["t_approx"], pred_boot, d_boot["event"])
        c_orig = concordance_index(d_full["t_approx"], pred_orig, d_full["event"])
        opt = c_boot - c_orig
        optimisms.append(opt)

        row_rhat, row_div = "", ""
        try:
            summ = az.summary(idata_b, var_names=["shape_k", "mu_approach"])
            row_rhat = float(summ["r_hat"].max())
            row_div = int(idata_b.sample_stats["diverging"].sum())
            max_rhats.append(row_rhat)
            n_divergences.append(row_div)
        except Exception:
            pass

        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(f"{b},{opt},{row_rhat},{row_div}\n")

        if (b + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  {b+1}/{B} done, {elapsed:.0f}s elapsed, ~{elapsed/(b+1-start_b)*(B-b-1):.0f}s remaining, "
                  f"running mean optimism={np.mean(optimisms):.4f}", flush=True)

    corrected = c_app - float(np.mean(optimisms))
    dist = c_app - np.array(optimisms)
    lo, hi = np.percentile(dist, [2.5, 97.5])

    result = dict(
        model="primary_hierarchical_bayesian_weibull_aft", apparent_c_index=float(c_app),
        bootstrap_corrected_c_index=float(corrected), ci_95_low=float(lo), ci_95_high=float(hi),
        n_resamples_used=len(optimisms), B=B,
        mean_max_rhat=float(np.mean(max_rhats)) if max_rhats else None,
        mean_n_divergences=float(np.mean(n_divergences)) if n_divergences else None,
        method_note="Bootstrap resamples fit via the SAME full 4-chain 2000-tune+2000-draw MCMC procedure "
                     "as the apparent model (no MAP shortcut) -- methodologically identical to the apparent "
                     "fit and to how the other 4 comparators were bootstrapped. B=100, not 500 (the other 4 "
                     "comparators used 500): full-MCMC-per-resample was calibrated at ~73s/resample "
                     "(~10.1h for B=500); B=100 (~2h) keeps the fitting METHOD identical across all 5 models "
                     "while keeping wall-clock time reasonable. This interval is consequently noisier "
                     "(wider/less stable) than the B=500 comparators' intervals -- stated explicitly, not "
                     "hidden. See module docstring and reports/head_a_validation_report.md.",
    )
    print("\nFINAL:", result)
    import json
    result = json.loads(json.dumps(result, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    with open("reports/head_a_bootstrap_bayesian.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print("wrote reports/head_a_bootstrap_bayesian.yaml")


if __name__ == "__main__":
    run()
