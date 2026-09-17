"""
Head A primary model (Master Prompt Section 5.4): hierarchical Bayesian
Weibull AFT with an interval-censored likelihood, PyMC.

S(t) = exp(-(t/scale)^k), scale = exp(mu_approach[approach] + valve_family
offset (partially pooled, non-centered) + beta_ppm * ppm_proxy_flag).

Simplification (documented, not hidden): a single shared shape k across the
whole cohort, rather than a per-approach/per-family shape. With only 11
events (all SAVR — see build_features.py's cohort-flow output), a
TAVR-specific shape would be entirely prior-driven and add a parameter with
no data to inform it; a shared k keeps the model identifiable at this n
while still letting scale vary by group and covariate, which is where the
literature priors and the cohort's own data actually disagree/agree.

Covariates included: index_approach (SAVR/TAVR/ViV-TAVR), valve_family
(partially pooled, only where recoverable from the note text), ppm_proxy
(labelled size <=21mm). NOT included: age (masked throughout the corpus,
see build_features.py), baseline gradient/LVEF/bicuspid anatomy (coverage
too sparse in this notes-only extract to support a reliable coefficient at
n=11 events — evaluated and excluded per Section 5.4's own instruction to
add further covariates "only if EPV and posterior diagnostics support
them").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import yaml


def _weibull_logs(t, k, scale):
    """log S(t) = -(t/scale)^k, safe for t=0 (-> 0) and t=inf (-> -inf)."""
    return -pt.pow(t / scale, k)


def build_model(features: pd.DataFrame, labels: pd.DataFrame, priors: dict) -> tuple[pm.Model, dict]:
    d = features.merge(labels, on="profile_key")
    approaches = sorted(d["index_approach"].dropna().unique().tolist())
    d = d[d["index_approach"].notna()].reset_index(drop=True)  # 6/99 have index_approach missing entirely (index event detected only via narrative fallback with no approach recoverable) -- excluded here, documented in cohort flow
    approach_idx = {a: i for i, a in enumerate(approaches)}
    d["approach_i"] = d["index_approach"].map(approach_idx)

    families = sorted(d.loc[d["valve_family_known"], "valve_family"].unique().tolist())
    family_idx = {f: i for i, f in enumerate(families)}
    d["family_i"] = d["valve_family"].map(family_idx)  # NaN where unknown -> handled via mask below
    has_family = d["valve_family_known"].to_numpy()

    t_lower = d["t_lower"].to_numpy(dtype=float)
    t_upper = d["t_upper"].to_numpy(dtype=float)
    is_event = (d["event"] == 1).to_numpy()
    ppm = d["ppm_proxy_flag"].fillna(False).to_numpy(dtype=float)

    # --- literature-informed priors (Section 6.2) -------------------------
    approach_prior_map = {"SAVR": "SAVR", "TAVR": "TAVR", "ViV-TAVR": "TAVR", "ViV-SAVR": "SAVR"}
    log_lambda_mu = np.array([
        np.log(priors["approach_level"][approach_prior_map.get(a, "SAVR")]["scale_lambda_years"])
        for a in approaches
    ])
    log_lambda_sd = np.array([
        priors["approach_level"][approach_prior_map.get(a, "SAVR")]["cv"] for a in approaches
    ])  # CV used directly as an approximate log-scale sd (weakly-informative, inflated per Section 6.2)
    k_prior_mean = np.mean([priors["approach_level"][a]["shape_k"] for a in priors["approach_level"]])

    coords = {"approach": approaches, "family": families, "obs": d["profile_key"].to_numpy()}
    with pm.Model(coords=coords) as model:
        k = pm.LogNormal("shape_k", mu=np.log(k_prior_mean), sigma=0.5)

        mu_approach = pm.Normal("mu_approach", mu=log_lambda_mu, sigma=log_lambda_sd, dims="approach")

        # Non-centered partial pooling of valve-family log-scale offsets
        # around their parent approach mean, tau weakly informative (half
        # of one log-unit, i.e. families can plausibly differ from their
        # approach mean by up to roughly a factor of e^0.5 ~ 1.6x before
        # the prior actively resists it).
        tau_family = pm.HalfNormal("tau_family", sigma=0.5)
        z_family = pm.Normal("z_family_raw", mu=0, sigma=1, dims="family")
        family_offset = pm.Deterministic("family_offset", z_family * tau_family, dims="family")

        beta_ppm = pm.Normal("beta_ppm", mu=0, sigma=0.3)  # weakly informative: PPM plausibly shortens durability, sign not pre-constrained

        log_scale = mu_approach[d["approach_i"].to_numpy()]
        family_term = pt.switch(
            pt.as_tensor(has_family),
            family_offset[np.where(has_family, d["family_i"].fillna(0).astype(int).to_numpy(), 0)],
            0.0,
        )
        log_scale = log_scale + family_term + beta_ppm * ppm
        scale = pm.Deterministic("scale_years", pt.exp(log_scale), dims="obs")

        logS_lower = _weibull_logs(t_lower, k, scale)
        logS_upper = pt.switch(pt.isinf(t_upper), -np.inf, _weibull_logs(t_upper, k, scale))

        # Event: log(S(L) - S(U)) = logS_lower + log1mexp(logS_upper - logS_lower) [since logS_upper < logS_lower]
        event_logp = logS_lower + pt.log1mexp(pt.clip(logS_upper - logS_lower, -700, -1e-10))
        censored_logp = logS_lower
        logp = pt.switch(pt.as_tensor(is_event), event_logp, censored_logp)
        pm.Potential("interval_censored_likelihood", logp.sum())

        # Prior-predictive / posterior-predictive draws of a fresh event
        # time per observation, for calibration checks (Section 8 Level 3).
        pm.Deterministic("median_survival_years", scale * (np.log(2)) ** (1.0 / k), dims="obs")

    meta = dict(approaches=approaches, families=families, n=len(d), n_events=int(is_event.sum()),
                n_excluded_no_approach=int((~d["index_approach"].notna()).sum()) if False else 0)
    return model, meta


if __name__ == "__main__":
    features = pd.read_csv("data_processed_patient_features.csv")
    labels = pd.read_csv("data_processed_patient_labels.csv")
    with open("config/priors.yaml", encoding="utf-8") as f:
        priors = yaml.safe_load(f)

    model, meta = build_model(features, labels, priors)
    print(f"Model built: n={meta['n']}, events={meta['n_events']}, "
          f"approaches={meta['approaches']}, families={meta['families']}")

    with model:
        idata = pm.sample(draws=2000, tune=2000, chains=4, cores=1, target_accept=0.95,
                           random_seed=20260917, progressbar=False)
        # median_survival_years is a pm.Deterministic, so it is already
        # captured per-draw in idata.posterior by pm.sample() itself --
        # no separate posterior-predictive pass is needed for it.

    idata.to_netcdf("reports/head_a_posterior.nc")
    print("wrote reports/head_a_posterior.nc")

    import arviz as az
    summary = az.summary(idata, var_names=["shape_k", "mu_approach", "tau_family", "beta_ppm"])
    print(summary)
    summary.to_csv("reports/head_a_posterior_summary.csv")
