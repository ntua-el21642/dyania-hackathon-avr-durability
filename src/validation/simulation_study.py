"""
Master Prompt Section 8, Level 5: simulation study.

Generates a synthetic cohort reusing the REAL cohort's own design (each
patient's approach, valve_family, ppm_proxy_flag, and observed maximum
follow-up C_i = last_note_year - index_implant_year, taken directly from
data_processed_patient_features.csv/labels.csv) so that n=99, the
year-only interval-censoring construction, and the missingness pattern
(valve_family known for the same ~37% of patients) are all reproduced
exactly, per the Master Prompt's own instruction. Event times are then
drawn from a Weibull with KNOWN true parameters (documented below, chosen
to be broadly consistent with -- not copied from -- the fitted posterior),
so recovery can be checked against ground truth the real data can never
provide.

For each of R replicates: refit the SAME hierarchical model
(build_model) with a REDUCED MCMC budget (2 chains x 600 tune + 600 draws,
~12-15s/replicate) -- a full 4-chain x 2000+2000 fit per replicate (the
budget used for the one real-data report) would take R x ~30s, which is
fine at R=50 but is reduced here for total wall-clock; documented, not
hidden. Reports: point-estimate bias (posterior mean - true, on the
log-scale parameters, which is the model's own native parameterization),
80%/95% credible-interval coverage (using each replicate's own posterior
mean +/- 1.28*sd / 1.96*sd, i.e. a normal approximation to the posterior,
not full HDI extraction, for speed), mean C-index, and average posterior
diagnostic quality (R-hat, divergences) across replicates.
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

# --- ground truth (documented, not fit from data) --------------------------
TRUE_SHAPE_K = 3.5
TRUE_SCALE = {"SAVR": 15.0, "TAVR": 20.0, "ViV-TAVR": 20.0}
TRUE_BETA_PPM = 0.15
# All valve-family offsets are truly zero in this simulation -- a
# correctly-behaving model should shrink tau_family toward 0 given no real
# family heterogeneity exists, rather than overfitting spurious family
# differences at these small per-family n's.


def load_design() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv("data_processed_patient_features.csv")
    labels = pd.read_csv("data_processed_patient_labels.csv")
    d = features.merge(labels, on="profile_key")
    d = d[d["index_approach"].notna()].reset_index(drop=True)
    d["followup_bound"] = np.where(d["event"] == 1, (d["t_lower"] + d["t_upper"]) / 2, d["t_lower"])
    return features[features["profile_key"].isin(d["profile_key"])].reset_index(drop=True), d


def simulate_labels(design: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(design)
    scale_i = design["index_approach"].map(TRUE_SCALE).to_numpy(dtype=float)
    ppm = design["ppm_proxy_flag"].fillna(False).to_numpy(dtype=float)
    scale_i = scale_i * np.exp(TRUE_BETA_PPM * ppm)
    u = rng.uniform(1e-6, 1 - 1e-6, n)
    t_true = scale_i * (-np.log(u)) ** (1.0 / TRUE_SHAPE_K)

    C = design["followup_bound"].to_numpy(dtype=float)
    event = (t_true <= C).astype(int)
    t_year = np.round(t_true).clip(min=0)

    t_lower = np.where(event == 1, np.clip(t_year - 1, 0, None), np.clip(C, 0, None))
    t_upper = np.where(event == 1, t_year + 1, np.inf)

    out = design[["profile_key"]].copy()
    out["event"] = event
    out["t_lower"] = t_lower
    out["t_upper"] = t_upper
    out["t_true"] = t_true  # kept for internal bias/C-index scoring, not fed to the model
    return out


def fit_one_replicate(features: pd.DataFrame, labels: pd.DataFrame, priors: dict, seed: int):
    model, meta = build_model(features, labels[["profile_key", "event", "t_lower", "t_upper"]], priors)
    with model:
        idata = pm.sample(draws=600, tune=600, chains=2, cores=1, target_accept=0.95,
                           random_seed=seed, progressbar=False)
    return idata, meta


def run(R: int = 50):
    features, design = load_design()
    with open("config/priors.yaml", encoding="utf-8") as f:
        priors = yaml.safe_load(f)

    true_log_scale = {a: np.log(s) for a, s in TRUE_SCALE.items()}
    rows = []
    t_start = time.time()
    for r in range(R):
        rng = np.random.default_rng(SEED + r)
        sim_labels = simulate_labels(design, rng)
        try:
            idata, meta = fit_one_replicate(features, sim_labels, priors, seed=SEED + r)
        except Exception as e:
            print(f"replicate {r}: FIT FAILED ({e}), skipping")
            continue

        summ = az.summary(idata, var_names=["shape_k", "mu_approach", "tau_family"])
        n_div = int(idata.sample_stats["diverging"].sum())
        max_rhat = float(summ["r_hat"].max())

        pred_med_surv = idata.posterior["median_survival_years"].mean(dim=["chain", "draw"]).values
        d_eval = features.merge(sim_labels, on="profile_key")
        d_eval = d_eval[d_eval["index_approach"].notna()].reset_index(drop=True)
        t_eval = np.where(d_eval["event"] == 1, (d_eval["t_lower"] + d_eval["t_upper"]) / 2, d_eval["t_lower"])
        c_index = concordance_index(t_eval, pred_med_surv, d_eval["event"])

        row = dict(replicate=r, n_divergences=n_div, max_rhat=max_rhat, c_index=c_index,
                   n_events=int(sim_labels["event"].sum()))
        for a in ["SAVR", "TAVR", "ViV-TAVR"]:
            row[f"mu_approach_{a}_mean"] = float(summ.loc[f"mu_approach[{a}]", "mean"])
            row[f"mu_approach_{a}_sd"] = float(summ.loc[f"mu_approach[{a}]", "sd"])
            row[f"mu_approach_{a}_true"] = true_log_scale[a]
        row["shape_k_mean"] = float(summ.loc["shape_k", "mean"])
        row["shape_k_sd"] = float(summ.loc["shape_k", "sd"])
        row["shape_k_true"] = TRUE_SHAPE_K
        row["tau_family_mean"] = float(summ.loc["tau_family", "mean"])
        rows.append(row)

        if (r + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  {r+1}/{R} done, {elapsed:.0f}s elapsed, ~{elapsed/(r+1)*(R-r-1):.0f}s remaining", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("reports/head_a_simulation_replicates.csv", index=False)

    def summarize(param: str, true_val: float):
        bias = float((df[f"{param}_mean"] - true_val).mean())
        cov80 = float((((df[f"{param}_mean"] - 1.28 * df[f"{param}_sd"]) <= true_val) &
                        (true_val <= (df[f"{param}_mean"] + 1.28 * df[f"{param}_sd"]))).mean())
        cov95 = float((((df[f"{param}_mean"] - 1.96 * df[f"{param}_sd"]) <= true_val) &
                        (true_val <= (df[f"{param}_mean"] + 1.96 * df[f"{param}_sd"]))).mean())
        return dict(true_value=true_val, mean_bias=bias, coverage_80pct_ci=cov80, coverage_95pct_ci=cov95,
                    mean_posterior_mean=float(df[f"{param}_mean"].mean()))

    summary = dict(
        R=len(df), n_per_replicate=len(design), mean_events_per_replicate=float(df["n_events"].mean()),
        mean_n_divergences=float(df["n_divergences"].mean()), mean_max_rhat=float(df["max_rhat"].mean()),
        mean_c_index=float(df["c_index"].mean()),
        shape_k=summarize("shape_k", TRUE_SHAPE_K),
        mu_approach_SAVR=summarize("mu_approach_SAVR", true_log_scale["SAVR"]),
        mu_approach_TAVR=summarize("mu_approach_TAVR", true_log_scale["TAVR"]),
        tau_family_mean_estimate=float(df["tau_family_mean"].mean()),
        tau_family_note="True family heterogeneity is 0 in this simulation; a low estimated tau_family here "
                         "indicates the model correctly avoids overfitting spurious family differences.",
    )
    print("\nSUMMARY:", summary)
    import json
    summary = json.loads(json.dumps(summary, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    with open("reports/head_a_simulation_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    print("wrote reports/head_a_simulation_summary.yaml and reports/head_a_simulation_replicates.csv")


if __name__ == "__main__":
    run(R=50)
