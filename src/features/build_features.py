"""
Builds patient_features.csv (landmark, index-implant-time covariates) and
patient_labels.csv (interval-censored time-to-event target) for Head A from
labels.csv, per Master Prompt Section 5.2/5.4.

Landmark = the index implant event itself (age/sex/approach/valve
model/size — everything in Section 5.1's extractor list #1-5 is, by
construction, known at implant, i.e. always <=landmark; there is no
separate "30-day post-implant echo" landmark available in this notes-only
extract, since baseline echo is recoverable for only a minority of
patients — see data_plan.md). No post-landmark note is ever read for a
FEATURE here — only for the label (redo_note_year / last_note_year), which
is the leakage boundary Hard Rule #2 requires.

Cohort restriction (documented, not silently applied): patients with
index_implant_source == 'undated' (18/117) have no time origin at all and
are EXCLUDED from Head A — a durability model cannot place them on a time
axis. This is reported, not hidden (see reports/head_a_cohort_flow.md).

Interval censoring (Master Prompt Section 0/8): year-only dates mean every
event time is only known to fall in [T-1, T+1] where T = event_note_year -
index_implant_year. Right-censored patients contribute their observed
follow-up time (last_note_year - index_implant_year) as a lower bound only.
"""
from __future__ import annotations

import pandas as pd
import yaml

with open("config/priors.yaml", encoding="utf-8") as f:
    PRIORS = yaml.safe_load(f)
VALVE_MODEL_TO_FAMILY = PRIORS["valve_model_to_family"]

# PPM proxy (Section 5.1, extractor #8): true iEOA requires BSA, which is
# not recoverable from these notes; Master Prompt explicitly specifies the
# documented proxy "labelled size <=21mm", flagged as a proxy not a
# measured value.
PPM_PROXY_SIZE_MM = 21


def build_patient_features(labels_path: str = "labels.csv") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(labels_path)
    flow = {"total_patients": len(df)}

    has_index_time = df["index_implant_source"] != "undated"
    flow["excluded_no_index_time"] = int((~has_index_time).sum())
    cohort = df[has_index_time].copy()
    flow["head_a_cohort"] = len(cohort)

    cohort["valve_family"] = cohort["index_valve_model"].map(VALVE_MODEL_TO_FAMILY)
    # Unmapped model (incl. Trifecta, Biocor, Epic, Konect, homograft, or
    # model never recovered at all) falls back to the approach-level
    # population prior at the modeling stage — never silently dropped, and
    # never assigned a fabricated family.
    cohort["valve_family_known"] = cohort["valve_family"].notna()
    cohort["ppm_proxy_flag"] = cohort["index_valve_size_mm"] <= PPM_PROXY_SIZE_MM
    cohort["size_known"] = cohort["index_valve_size_mm"].notna()

    # --- time-to-event construction -----------------------------------
    # PRIMARY definition: event = bvf_stage==2 AND NOT
    # pending_physician_reconfirmation. 4/117 patients (058, 061, 068, 081)
    # have real textual reintervention evidence but a disagreeing,
    # unresolved physician blind-adjudication determination (see
    # config/label_overrides.yaml, review/error_catalogue.md) -- team
    # decision 2026-09-17: do not count them as confirmed events for the
    # primary analysis while reconfirmation is pending; they are censored
    # at their last observed note year instead, and re-run as events only
    # in the SENSITIVITY definition below (which doubles as the Master
    # Prompt's own "definite-only vs. definite+probable" sensitivity
    # check, Section 8 Level 5).
    if "pending_physician_reconfirmation" not in cohort.columns:
        cohort["pending_physician_reconfirmation"] = False
    is_pending = cohort["pending_physician_reconfirmation"].fillna(False).astype(bool)
    is_event_primary = (cohort["bvf_stage"] == 2) & (~is_pending)
    is_event_sensitivity = cohort["bvf_stage"] == 2  # includes the 4 pending patients as events

    t_event = cohort["redo_note_year"] - cohort["index_implant_year"]
    t_censor = cohort["last_note_year"] - cohort["index_implant_year"]

    def _build_time_cols(is_event):
        # Interval-censored bounds in years since index implant. Event:
        # [T-1, T+1], clipped at 0. Censored: lower bound = observed
        # follow-up, upper bound = +inf.
        t_lower = pd.Series(index=cohort.index, dtype=float)
        t_upper = pd.Series(index=cohort.index, dtype=float)
        t_lower[is_event] = (t_event[is_event] - 1).clip(lower=0)
        t_upper[is_event] = t_event[is_event] + 1
        t_lower[~is_event] = t_censor[~is_event].clip(lower=0)
        t_upper[~is_event] = float("inf")
        return t_lower, t_upper

    cohort["event"] = is_event_primary.astype(int)
    cohort["t_lower"], cohort["t_upper"] = _build_time_cols(is_event_primary)
    cohort["event_sensitivity_incl_pending"] = is_event_sensitivity.astype(int)
    cohort["t_lower_sensitivity"], cohort["t_upper_sensitivity"] = _build_time_cols(is_event_sensitivity)

    feature_cols = [
        "profile_key", "index_implant_year", "index_implant_source", "index_approach",
        "index_valve_model", "valve_family", "valve_family_known",
        "index_valve_size_mm", "size_known", "ppm_proxy_flag",
    ]
    label_cols = ["profile_key", "event", "t_lower", "t_upper",
                  "event_sensitivity_incl_pending", "t_lower_sensitivity", "t_upper_sensitivity",
                  "confidence_tier", "hvd_stage", "bvf_stage", "pending_physician_reconfirmation"]

    features_df = cohort[feature_cols].reset_index(drop=True)
    labels_out_df = cohort[label_cols].reset_index(drop=True)

    flow["events_primary_confirmed_only"] = int(cohort["event"].sum())
    flow["events_sensitivity_incl_pending"] = int(cohort["event_sensitivity_incl_pending"].sum())
    flow["n_pending_physician_reconfirmation"] = int(is_pending.sum())
    flow["pending_patients"] = cohort.loc[is_pending, "profile_key"].tolist()
    flow["censored"] = int((cohort["event"] == 0).sum())
    flow["events_by_approach"] = cohort.loc[cohort["event"] == 1, "index_approach"].value_counts().to_dict()
    flow["missing_valve_family"] = int((~cohort["valve_family_known"]).sum())
    flow["missing_size"] = int((~cohort["size_known"]).sum())
    flow["age_available"] = False  # [AGE] is masked throughout the corpus — see note below
    flow["age_note"] = (
        "Age at implant is masked ([AGE]) in every note in this corpus with no numeric value "
        "surviving de-identification anywhere it was checked; age cannot be used as a covariate "
        "or as a comparator model (Master Prompt's own 'age-only Weibull' comparator, Section 5.4, "
        "cannot be built for this reason — documented as a limitation, not silently substituted)."
    )
    return features_df, labels_out_df, flow


if __name__ == "__main__":
    features_df, labels_df, flow = build_patient_features()
    features_df.to_csv("data_processed_patient_features.csv", index=False)
    labels_df.to_csv("data_processed_patient_labels.csv", index=False)
    print(yaml.safe_dump(flow, sort_keys=False, allow_unicode=True))
