# Model Approach

---

## 1. Problem Formulation

Two separate prediction tasks, deliberately not merged into one:

- **Head B — severity staging (deterministic, rule-based):** given a patient's notes up to the current point in
  time, assign a VARC-3 hemodynamic-valve-deterioration (HVD) stage (0/1/2/3), bioprosthetic-valve-failure (BVF)
  stage (0/2/3), an S/R/RS phenotype, and a confidence tier (definite/probable/possible/excluded/censored). This is
  a rule engine, not a trained classifier — see §2 for why.
- **Head A — durability forecast (probabilistic, time-to-event):** given landmark (index-implant-time) covariates,
  predict the distribution of time from implant to bioprosthetic valve failure (BVF Stage ≥2: reintervention).
  Framed as parametric survival analysis (Weibull AFT) with an **interval-censored** likelihood, not a fixed-horizon
  binary classifier, because (a) our dates are year-only, so every event time is only known to fall within ±1 year
  of the observed year (Master Prompt Hard Rules §0), and (b) a survival framing lets us report calibrated,
  horizon-flexible risk (e.g. P(BVF) at 5 vs. 10 years) rather than a single arbitrary cutoff, and degrades honestly
  to wide uncertainty rather than a false point estimate when data is sparse — which it is here.

> **Event count, final (2026-09-17, physician sign-off received):** full-cohort physician blind adjudication
> disagreed with the pipeline on 11/117 patients (`review/error_catalogue.md` §8). Mechanical re-verification
> against raw note text and one confirmed code fix resolved 7 of those directly. The remaining 4 (058, 061, 068,
> 081) required a physician judgment call and were held as `probable`/`pending_physician_reconfirmation=True`
> pending sign-off; **that sign-off was received 2026-09-17, asymmetrically**: 058, 061, and 068 confirmed the
> pipeline's original `definite` read was correct (no label change), while 081 confirmed the *physician's* original
> read over the pipeline's — the pipeline's automated BVF Stage 2 call for 081 is a genuine over-call, documented as
> a known, not-yet-fixed extractor gap in `config/label_overrides.yaml` (`extract_implicit_new_tavr` needs a
> corroborating-finding requirement it currently lacks). **Final confirmed event count: 8/117** (`bvf_stage==2`, all
> `confidence_tier=definite`) — zero patients remain `pending_physician_reconfirmation`. The pipeline still emits a
> dual-column schema (`event` / `event_sensitivity_incl_pending`) as a methodological safeguard, but the two now
> coincide at 8 since nothing is pending anymore; the primary-vs-sensitivity distinction from earlier drafts no
> longer applies. Every "N events" mention below refers to this final 8-event count.

Competing risk of death is *not* modeled as a separate cause-specific hazard in the current build — no structured
vital-status field survives de-identification in this notes-only extract (documented in `varc3_rules.py`'s BVF
Stage 3 comment), so death is neither observed nor derivable. This is a real limitation, not an oversight: any
patient who died without a documented reintervention is indistinguishable, in this data, from a patient who is
simply right-censored at their last note. Section 7 below states this in every report we'd generate.

## 2. Chosen Model(s)

| Model | Role | Justification |
|---|---|---|
| Rule-based VARC-3 engine (Head B) | Primary severity classifier | Full auditability required for Level-1 physician validation (every label traces to a source sentence); also the label source for Head A, so its own precision had to be established first — see `review/error_catalogue.md`. |
| Hierarchical Bayesian Weibull AFT, PyMC (Head A) | Primary durability model | Handles interval-censored + right-censored data natively via a custom likelihood; AFT parameterization gives directly interpretable "time ratios" (years gained/lost); partial pooling + literature-informed priors let a handful of confirmed events (8, final physician-confirmed count) borrow strength from published cohorts honestly, with the prior/posterior disagreement itself reported as a finding. |
| Literature-prior-only (null) | Comparator | What you'd predict with *zero* cohort data — the floor the fitted model must beat. |
| Valve-family-only Weibull (frequentist MLE, `lifelines.WeibullFitter`) | Comparator | Simplest possible data-driven baseline; shows directly that 2 of 3 observed valve families have *zero* events (MLE undefined) — a data-sparsity fact the primary model's partial pooling is specifically designed to handle gracefully. |
| Penalized Cox, elastic net (`lifelines.CoxPHFitter(penalizer=0.1, l1_ratio=0.5)`) | Comparator | Master Prompt specifies scikit-survival; that package's `ecos` dependency failed to build on this Windows environment (no MSVC C++ Build Tools installed — `pip install scikit-survival` error confirmed and logged). Substituted with lifelines' equivalent elastic-net-penalized Cox partial likelihood, which needs no C compiler. **Confirmed, not assumed:** `cph.penalizer == 0.1` and `cph.l1_ratio == 0.5` verified directly on the fitted object, and — more concretely — an **unpenalized** Cox fit (`penalizer=0.0`) on the identical covariates fails outright with `ConvergenceError: Matrix is singular` (`ppm_proxy_flag` triggers a near-complete-separation warning at this event count). The elastic-net penalty is not an optional refinement here; it is the only reason any Cox model can be fit on this data at all. |
| XGBoost AFT (`survival:aft`, interval-censored labels) | Comparator / deployment-scale candidate | Scalable production candidate per Section 5.4; explicitly *not* the reported model at this n. |
| PyTorch DeepSurv/DeepHit | Not built | Explicitly not justified at n=117/8 confirmed events per Master Prompt Section 5.4; documented as a future, deployment-scale option only. |

## 3. Feature Engineering

### Input Variables (landmark = index implant event; Section 5.1, extractors #1–16)

Recovered for the full 117-patient cohort from notes alone: approach (SAVR/TAVR/ViV — recovered for 99/117 after
fixing the newline-wrap bug described below), valve model/family (58/117), labelled size in mm (58/117), redo/ViV
reintervention status with negation handling, explicit SVD/BVF language, morphological findings (leaflet
thickening/calcification/restriction), AR/PVL grade, and exclusion terms (endocarditis, thrombosis, PVL). **Age at
implant is not recoverable at all** — `[AGE]` is masked in every note with zero surviving numeric values anywhere in
the 215-note corpus (verified by a full-corpus regex scan) — so no age covariate, and no "age-only" comparator,
appear anywhere in this build. **Confirmed directly against the code and data, not just stated in prose:**
`data_processed_patient_features.csv` has no age column at all — not blank, not zero-filled, absent as a field —
and `weibull_aft_bayes.build_model()` never takes an age input anywhere in the likelihood construction (see its
covariate list: `index_approach`, `valve_family`, `ppm_proxy_flag` only). Age was never a covariate in any version
of the Head A model reported in this repo; it is absent by construction, not removed after being present. LVEF, bicuspid anatomy, and baseline gradient/EOA/DVI are extractable but too sparse
across the cohort to support a reliable coefficient at 8 confirmed events; evaluated and excluded from Head A per Section
5.4's own instruction to add further covariates "only if EPV and posterior diagnostics support them."

### Engineered Features

| Feature | Derivation | Clinical Rationale |
|---|---|---|
| `valve_family` | Regex/dictionary match on valve model name → mapped to one of the literature-prior families in `config/priors.yaml`; unmapped (incl. Trifecta — see §5) falls back to the approach-level prior | Different valve families have materially different published durability curves (e.g. Mitroflow vs. Hancock II); the hierarchy lets each family have its own scale while still being partially pooled toward its approach when data is sparse. |
| `ppm_proxy_flag` | `index_valve_size_mm <= 21` | True effective-orifice-area-indexed PPM requires BSA, unavailable here; Master Prompt Section 5.1 #8 specifies this exact proxy, flagged as a proxy throughout. |
| `event` / `t_lower` / `t_upper` | `event = (bvf_stage==2)`; event interval = `[redo_year - index_year - 1, redo_year - index_year + 1]` clipped at 0; censored lower bound = `last_note_year - index_year`, upper bound = ∞ | Direct implementation of the year-only interval-censoring rule (Master Prompt §0). |

### Handling Missing Data

- **Cohort restriction, not imputation:** 18/117 patients have no recoverable index-implant year at all
  (`index_implant_source == 'undated'`) and are excluded from Head A — there is no time origin to place them on.
  Reported explicitly (`src/features/build_features.py`'s cohort-flow output), not silently dropped.
- **Valve family unknown (62/99 in the Head A cohort):** not imputed — these patients keep their approach-level
  prior only (no family-level offset applied), which is exactly what the hierarchical structure is for.
- **No multiple imputation, no SMOTE/oversampling** anywhere in this build, per Hard Rule #6 — missingness here is
  block-wise/by-extraction-design (labs/meds only exist for 17/117 patients; valve model name only survives
  de-identification/dictionary-matching for about half the cohort), not something a random-draw imputation model
  should paper over.
- **Informative missingness is flagged, not filled:** `valve_family_known` and `size_known` indicator columns are
  retained in `data_processed_patient_features.csv` precisely so a reader can see what was and wasn't recoverable,
  rather than a silently-imputed value masquerading as observed data.

## 4. Validation Strategy

- **No train/val/test split.** At n=99/8 confirmed events a hold-out split would leave single-digit events in each arm,
  which is not a meaningful test. Master Prompt Section 8 Level 3 specifies Harrell bootstrap optimism correction
  instead — run on the final 8-event cohort (see `reports/head_a_validation_report.md` for the current numbers:
  Level 1 label validity, Level 2 construct validity, Level 3 bootstrap-corrected comparator C-indices, Level 4
  leakage controls, Level 5 simulation study).
- **Patient-level integrity:** every unit here (bootstrap resample, permutation shuffle) operates on whole patients;
  there is no note-level or note-fragment splitting anywhere in the pipeline.
- **Cross-validation:** not applicable in the same sense — the primary model is a single Bayesian fit on the full
  cohort with informative priors substituting for a training/validation split; the frequentist comparators
  (Cox, XGBoost AFT) are reported in-sample only, with that limitation stated plainly in `comparators.py`'s own
  docstring and in the validation report.
- **Temporal / external validation:** not applicable at this stage (single-center, single extract); documented as
  future work in `data_plan.md`.

## 5. Priors (Section 6.1–6.2)

Every literature anchor in `config/priors.yaml` was **re-verified directly against the source PDFs** on
2026-09-17 (`papers/ejcts_52_3_408.pdf`, `papers/JAH3-14-e041505.pdf`) — text-extracted page by page and checked
against the Master Prompt's own transcription rather than taken on faith. All values matched exactly. Weibull
(shape k, scale λ) priors were fit by least-squares on `log(-log S(t))` vs. `log(t)` per Section 6.2; single-point
families borrow shape k from their approach-level population fit (documented, with an inflated CV to reflect the
extra uncertainty of a one-point fit).

**Trifecta gap:** confirmed by a full-text search of all three provided PDFs — zero matches for "Trifecta" in any
of them. Trifecta valves account for 3 of the 8 seed-table-confirmed SVD-signal patients in our own cohort (a known
clinical early-SVD signal for this device), but per Master Prompt's explicit instruction we do **not** invent a
Trifecta prior — it is assigned only the SAVR-pericardial population-level prior, with a note attached in
`priors.yaml` flagging this for closer physician surveillance pending a sourced estimate.

> **⚠ Family-level differences from this model — including anything about Trifecta — are NOT confirmed at this n,
> and must not be read as a finding.** The Level 5 simulation study (`reports/head_a_validation_report.md`)
> recovered a non-trivial family-pooling variance (`tau_family` ≈ 0.34) **even when the simulated ground truth had
> zero true family heterogeneity** — nearly identical to the real cohort's own fitted `tau_family` of 0.30. This
> means the model's own hierarchical structure produces apparent family-to-family spread as a baseline artifact of
> partial pooling at small per-family n, independent of whether real clinical heterogeneity exists. Every family-
> level number this model produces (the forest plot in §6, any Trifecta-specific discussion, any "family X differs
> from family Y" statement) should be read as **not distinguishable from this pooling artifact with the current
> event count**, not as evidence of a real family effect. This applies retroactively to every family-level mention
> elsewhere in this document and in `reports/head_a_validation_report.md`.

## 6. Expected Model Outputs

Per patient at the landmark: predicted median freedom-from-BVF time (years, posterior median + 89% credible
interval), P(BVF) at any horizon via the posterior survival function, and — once a risk-category cutoff is
calibrated on a larger/adjudicated cohort — a low/intermediate/high tier. Current build reports the posterior
distribution itself (`reports/head_a_posterior.nc`) plus per-group time ratios (`reports/head_a_forest_plot.png`).
**The family-level rows in that forest plot specifically are not confirmed findings — see the tau_family warning
in §5** — only the approach-level (SAVR vs. TAVR) rows have any real event count behind them. A per-patient
waterfall decomposition (cohort-median → individual prediction, one step per covariate) is designed but not yet
implemented as a standalone report generator.

## 7. Clinical Integration

Intended as a decision-support flag inside echo-surveillance scheduling: a patient trending toward the
high-uncertainty/short-median-durability end of Head A's posterior, or accumulating Head B possible/probable-tier
findings, triggers an earlier structural-heart-team referral or shortened surveillance interval — never an
automatic reintervention trigger. Every report a clinician would see states plainly that these are population-level
statistical associations, not individualized causal predictions, and that the underlying labels, while
physician-sign-off-confirmed on every adjudicated disagreement, still derive from a small, single-center,
notes-only extract (n=99 patients/8 events) rather than an independently validated ground truth.

## 8. Limitations and Failure Modes

- **n=99/8 confirmed events (Head A), final.** Every model here — including the primary one — should be read as
  demonstrating a *method*, not a validated clinical tool. **Bootstrap-corrected C-index with 95% CI, not the
  naive in-sample value** (Harrell optimism correction; full table in `reports/head_a_validation_report.md`
  Level 3, final 2026-09-17 on the physician-confirmed 8-event cohort): penalized Cox 0.546 apparent →
  **0.536 corrected, CI [0.467, 0.573]**; XGBoost AFT 0.507 → **0.497, CI [0.436, 0.513]** (essentially exact
  chance); valve-family-only Weibull degenerate (**[0.500, 0.500]**, 213/500 = 43% of resamples fail to fit at
  all); literature-prior-only (null, not bootstrapped) apparent 0.493. **Primary hierarchical Bayesian Weibull
  AFT: 0.618 apparent → 0.554 corrected, 95% CI [0.353, 0.771]** (B=100 full 4-chain MCMC per resample, same
  fitting procedure as its own apparent fit and as the other four comparators — B=100 vs. the others' B=500 is a
  stated compute-time tradeoff, not a method difference). In plain terms: in roughly 55 of 100 random pairwise
  patient comparisons, the model correctly ranked who failed first — barely above chance on its central estimate,
  with a 95% interval spanning worse-than-chance to fairly strong. Its point estimate remains the highest of the
  five models, but "beats chance with confidence" cannot be claimed at this n. A case-level check tells the same
  story concretely: of the 8 confirmed-event patients, only 3 had their actual event fall inside the model's own
  80% credible interval for predicted median survival time — consistent with the Level 5 simulation study's own
  under-coverage finding (interval coverage 48-56% vs. the nominal 80%), not a new surprise.
- **All 8 confirmed events are SAVR-index.** TAVR-index reinterventions are essentially unobserved in this
  notes-only extract; the TAVR-arm posterior is almost entirely prior-driven. This may reflect this being a younger
  technology with less mature follow-up in our corpus specifically, not a true absence of TAVR SVD risk.
- **Simulation study (Level 5, `reports/head_a_validation_report.md`) found real miscalibration, not a clean pass:**
  across 50 synthetic replicates with a known ground truth, `shape_k` was underestimated (bias -0.79) with 80%
  credible intervals covering the truth only 56% of the time (nominal 80%) — the shape prior pulls estimates down
  when the true shape exceeds the literature's own pooled average, and this sample size doesn't fully overcome
  that. More consequentially: **`tau_family` (family-level pooling variance) recovered to 0.34 even when the
  simulation's true family heterogeneity was exactly zero** — nearly identical to the real cohort's own fitted
  `tau_family` of 0.30. This means a meaningful share of the real model's apparent family-to-family durability
  differences may be an artifact of the pooling structure at small per-family n, not necessarily genuine clinical
  heterogeneity between valve families. Any family-level statement made from this model (e.g. "family X is less
  durable than family Y") should be treated as provisional until this is checked with more events per family.
- **Posterior durability estimates for this cohort run notably shorter than the general literature** (e.g. SAVR
  posterior median ≈14 years vs. the ~79%/49% freedom-from-SVD at 15/20y reported by Bourguignon et al. for a much
  larger general cohort). This is flagged, not smoothed over — it may reflect genuine case-mix concentration (this
  looks like a structural-heart referral center with a concentrated redo/ViV caseload) or the small-n/selection-bias
  limitations already noted in `data_plan.md`, and needs the physician team's read before being trusted.
- **scikit-survival substitution** (elastic-net Cox via lifelines instead) is an environment constraint, documented
  in `comparators.py`, not a methodological choice.
- **Guyot KM-curve reconstruction (Section 5.6) has no real data to run on.** All 3 provided source PDFs were
  full-text searched and confirmed to contain no actual KM curve with a numbers-at-risk table (one schematic
  figure, otherwise reporting-standard prose only) — per Hard Rule #1 we do not fabricate digitization
  coordinates. The algorithm itself is implemented and self-tested against synthetic data (max deviation 0.054
  vs. a known true curve, `src/external/guyot.py`), and a concrete checklist naming the 5 primary papers a team
  member would need to source exists (`data/external/km_digitized/DIGITIZATION_CHECKLIST.md`), so this is
  execution-ready the moment real digitized coordinates exist — just never pointed at real data in this build.
- **PyTensor's C compiler backend fails on this Windows environment** (missing MSVC Build Tools); the model runs
  correctly via the pure-Python fallback (`PYTENSOR_FLAGS="cxx="`), documented in `README.md`, at some speed cost
  (~30-40s for 4 chains × 4000 draws on this cohort size — immaterial at this n, but would not scale to a much
  larger cohort without either installing a compiler or moving to a cloud/Linux environment).
