"""
Master Prompt Section 8, Level 2 — construct / known-groups validity.

Checks whether the pipeline's own labels reproduce three associations the
literature would predict, using ONLY the Head A cohort (n=99, 11 events)
built earlier this session. Age cannot be checked at all (masked
throughout the corpus, zero surviving values — see build_features.py); this
is stated explicitly rather than silently skipped.

Per the Master Prompt: "Failure to reproduce a strong known association
triggers label review." Results here are reported as-is, including the
one association that does NOT clearly replicate (labelled size) and one
that is confounded rather than cleanly reproduced or refuted (Trifecta) —
neither is smoothed over.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats


def load_cohort() -> pd.DataFrame:
    features = pd.read_csv("data_processed_patient_features.csv")
    labels = pd.read_csv("data_processed_patient_labels.csv")
    d = features.merge(labels, on="profile_key")
    d = d[d["index_approach"].notna()].reset_index(drop=True)
    d["t_approx"] = d.apply(lambda r: (r["t_lower"] + r["t_upper"]) / 2 if r["event"] == 1 else r["t_lower"], axis=1)
    return d


def check_age() -> dict:
    return dict(
        checkable=False,
        reason="Age at implant is masked ([AGE]) throughout the corpus with zero surviving numeric values "
               "(verified by full-corpus regex scan, see build_features.py). The literature's 'younger age -> "
               "higher SVD' association cannot be tested against this cohort at all.",
    )


def check_labelled_size(d: pd.DataFrame) -> dict:
    ev = d.loc[d["event"] == 1, "index_valve_size_mm"].dropna()
    cn = d.loc[d["event"] == 0, "index_valve_size_mm"].dropna()
    out = dict(n_event=len(ev), n_censored=len(cn),
               mean_event=float(ev.mean()) if len(ev) else None,
               mean_censored=float(cn.mean()) if len(cn) else None)
    if len(ev) >= 2 and len(cn) >= 2:
        t, p_t = stats.ttest_ind(ev, cn, equal_var=False)
        u, p_u = stats.mannwhitneyu(ev, cn, alternative="two-sided")
        out.update(welch_t=float(t), welch_p=float(p_t), mannwhitney_p=float(p_u))
        out["replicated"] = bool(out["mean_event"] < out["mean_censored"] and p_t < 0.10)
    else:
        out["replicated"] = None
    return out


def check_valve_family_and_trifecta(d: pd.DataFrame) -> dict:
    sub = d[d["valve_family_known"]]
    family_table = sub.groupby("valve_family")["event"].agg(n="count", events="sum")
    family_table["rate"] = family_table["events"] / family_table["n"]

    model_table = d.groupby("index_valve_model")["event"].agg(n="count", events="sum")
    model_table["rate"] = model_table["events"] / model_table["n"]
    model_table["mean_followup_years"] = d.groupby("index_valve_model")["t_approx"].mean()

    trifecta_rate = model_table.loc[["Trifecta", "Trifecta GT"], "rate"].mean() if \
        {"Trifecta", "Trifecta GT"}.issubset(model_table.index) else None
    trifecta_followup = model_table.loc[["Trifecta", "Trifecta GT"], "mean_followup_years"].mean() if \
        {"Trifecta", "Trifecta GT"}.issubset(model_table.index) else None
    other_savr_followup = model_table.drop(index=["Trifecta", "Trifecta GT"], errors="ignore")
    other_savr_followup = other_savr_followup[other_savr_followup["mean_followup_years"] > 0]["mean_followup_years"].mean()

    return dict(
        family_table=family_table.to_dict(orient="index"),
        model_table=model_table.to_dict(orient="index"),
        trifecta_raw_rate=trifecta_rate,
        trifecta_mean_followup_years=trifecta_followup,
        other_models_mean_followup_years=float(other_savr_followup),
        interpretation=(
            "Trifecta/Trifecta GT's own raw event rate (5.6%/14.3%) is NOT higher than Biocor (100%, n=2) or "
            "Magna (33%, n=3) in this cohort's raw numbers, which on its face does not reproduce the literature's "
            "Trifecta early-SVD signal. But Trifecta/Trifecta GT patients have a much shorter mean observed "
            "follow-up (~3.5y) than Biocor/Magna (7-14.5y) in this cohort -- consistent with Trifecta being a "
            "newer device with less elapsed implant-to-now time in this specific snapshot, not necessarily lower "
            "true risk. Combined with the clean 'no SVD before 5 years' finding below (all 11 events occurred at "
            "5-15 years post-implant), most of this cohort's Trifecta patients (mean follow-up 3.5y) have not yet "
            "reached the window where SVD would typically appear. This is a CONFOUNDED comparison, not a clean "
            "replication or refutation -- reported as such, not smoothed into either direction."
        ),
    )


def check_early_svd_rarity(d: pd.DataFrame) -> dict:
    savr_events = d[(d["event"] == 1) & (d["index_approach"] == "SAVR")]
    t = savr_events["t_approx"]
    return dict(
        n_savr_events=len(t),
        n_before_5y=int((t < 5).sum()),
        min_years=float(t.min()), median_years=float(t.median()), max_years=float(t.max()),
        replicated=bool((t < 5).sum() == 0),
    )


if __name__ == "__main__":
    d = load_cohort()
    print(f"Cohort: n={len(d)}, events={int(d['event'].sum())}\n")

    print("=== Age at implant ===")
    print(check_age(), "\n")

    print("=== Labelled size (smaller -> higher SVD expected) ===")
    size_result = check_labelled_size(d)
    print(size_result, "\n")

    print("=== Valve family / Trifecta ===")
    fam_result = check_valve_family_and_trifecta(d)
    print("family_table:", fam_result["family_table"])
    print("model_table:", fam_result["model_table"])
    print("interpretation:", fam_result["interpretation"], "\n")

    print("=== SVD rarity before 5 years (SAVR) ===")
    early_result = check_early_svd_rarity(d)
    print(early_result)

    import json
    import yaml
    payload = json.loads(json.dumps(
        dict(age=check_age(), labelled_size=size_result, valve_family=fam_result, early_svd_rarity=early_result),
        default=float))
    with open("reports/head_a_level2_construct_validity.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print("\nwrote reports/head_a_level2_construct_validity.yaml")
