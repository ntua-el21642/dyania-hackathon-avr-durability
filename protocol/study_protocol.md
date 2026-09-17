# Study Protocol

---

## 1. Study Title and Objectives

**Title:** Notes-Derived Structural Valve Deterioration Staging and Bayesian Durability Forecasting for
Bioprosthetic Aortic Valve Recipients

**Primary Objective:** Given a patient's clinical notes up to and including the index bioprosthetic aortic valve
implant, forecast the time to bioprosthetic valve failure (BVF Stage ≥2: reintervention) and stage current
structural valve deterioration severity per VARC-3, with every assignment traceable to source text.

**Secondary Objectives:**
1. Differentiate durability trajectories by implant approach (SAVR vs. TAVR vs. valve-in-valve) and valve family.
2. Characterize the S/R/RS (stenosis/regurgitation/mixed) phenotype distribution of SVD in this cohort.
3. Quantify how much a literature-informed Bayesian prior improves discrimination over purely data-driven
   comparators at a sample size (n=99, 8 events) too small to support an unconstrained model.
4. Establish and validate a reproducible, auditable NLP labeling pipeline as the prerequisite for any of the above —
   in practice, this consumed the majority of the effort in this build, since a durability model is only as good as
   its labels (see `review/error_catalogue.md`).

## 2. Target Population

### Inclusion Criteria
Patients with a documented bioprosthetic aortic valve implant (SAVR or TAVR) with at least one note in the
extract dated at or after the index implant, and a recoverable index-implant year (source: Operative
Report/Procedures note, or a narrative implant mention with an explicit nearby date).

### Exclusion Criteria
- Mechanical valve recipients (out of scope — bioprosthetic durability only).
- Patients whose only documented reintervention/dysfunction is attributable to a non-structural competing event —
  endocarditis, valve thrombosis, paravalvular leak, or patient-prosthesis mismatch — per VARC-3 §4.4; these are
  labeled `excluded` (competing event), never folded into the SVD/BVF endpoint.
- Patients with no recoverable index-implant year (18/117 in this extract) — excluded from Head A specifically,
  since there is no time origin to place them on; they remain in the Head B (severity-only) cohort.
- Events or death occurring before the landmark — excluded from the prediction cohort by construction, since the
  landmark *is* the index implant event.

### Cohort Size Estimate
117 patients total (Head B / severity cohort); 99 with a Head A time origin, of whom 8 have a physician-confirmed
BVF Stage 2 event (final, post-sign-off — see §5 and `review/error_catalogue.md` §8). This is well below the
sample size a frequentist survival model would need for stable, generalizable
covariate estimates — the entire modeling strategy (hierarchical Bayesian partial pooling + literature-informed
priors, §5 of `model/approach.md`) is a direct response to this constraint, not an incidental choice. A future
multi-site extension would need on the order of several hundred confirmed events (guided by an EPV rule of
thumb of ≥10 events per covariate for the eventual unpooled model) to support the richer covariate set (LVEF,
bicuspid anatomy, baseline hemodynamics) that this build could not include for coverage reasons.

## 3. Endpoints

### Primary Endpoint
**BVF Stage ≥2** (aortic valve reintervention — redo surgery or valve-in-valve TAVR), assessed continuously via
notes review and staged per VARC-3 (Généreux et al., Eur Heart J 2021). Threshold for clinical usefulness: the
model's posterior time ratios and risk categories should meaningfully separate valve families/approaches with
already-known different durability profiles in the literature (a construct-validity check — Master Prompt §8 Level
2 — run on the final 8-event cohort; results are mixed/confounded rather than a clean pass, see
`reports/head_a_validation_report.md` Level 2 for the full breakdown).

### Secondary Endpoints

| Endpoint | Measurement | Timeframe |
|---|---|---|
| Moderate-or-greater HVD (expanded endpoint) | VARC-3 Stage 2 hemodynamic threshold (or Capodanno absolute-threshold fallback when no baseline echo exists) | Continuous, from implant |
| S/R/RS phenotype | Stenotic vs. regurgitant vs. mixed hemodynamic pattern at time of worst post-implant finding | At time of HVD stage assignment |
| Time-to-reintervention | Interval-censored years from index implant to redo/ViV note | Continuous |
| Mean-gradient progression | Extractable for ~50/117 patients (135 values); only 1 patient has ≥2 distinct-year gradients, so longitudinal VARC-3 Δ-gradient staging is the exception, not the norm | Where available |

## 4. Proposed Data Sources

See `data/data_plan.md` for the full table (real-deployment vs. hackathon-prototype sources) and preprocessing
detail.

**Ground Truth Definition:** Five-tier hierarchy per Master Prompt Section 5.3 — Definite (reintervention op note,
negation-checked, or explicit SVD/BVF text not confined to a bare coded problem-list entry) → Probable (hemodynamic
threshold met without a baseline reference echo) → Possible (morphology-only or qualitative-trend-only finding, or
a bare coded-history SVD phrase with unconfirmed etiology) → Excluded (endocarditis/thrombosis/PVL competing event)
→ Censored (no post-implant evidence). Implemented deterministically in `varc3_rules.py`, first validated against a
physician-reviewed 19-patient seed set (reintervention-detector F1 = 1.000 after four root-caused bug fixes), then
against a full 117-patient physician blind adjudication (`review/adjudication_form.xlsx`) — the blind full-cohort
read scored much lower (F1 = 0.353, near-chance kappa) than the anchored seed check, a discrepancy attributed to
confirmation bias in how the seed set was originally built (it reviewed automated hits rather than reading blind)
rather than to pipeline regression; see `reports/head_a_validation_report.md` Level 1 for the full discussion. Of
11 disagreements surfaced by the blind read, 7 were resolved by mechanical raw-text re-checks, code fixes, or a
definitional schema split (`any_AV_reintervention` vs. `SVD_specific_BVF`); the remaining 4 (Patient_058, 061, 068,
081) required physician judgment calls and were resolved via physician sign-off received 2026-09-17 (asymmetric
outcome: 058/061/068 confirmed the pipeline's original read, 081 confirmed the physician's read over the
pipeline's, which is documented as a known extractor gap in `config/label_overrides.yaml`). See
`review/error_catalogue.md` §8 for the full audit trail.

## 5. Statistical Analysis Plan

### Sample Size
**8 physician-confirmed primary-endpoint events** (all `confidence_tier=definite`, final — physician sign-off
received 2026-09-17 on the last four disputed cases; see `review/error_catalogue.md` §8 and
`config/label_overrides.yaml`), plus 22 expanded-endpoint-only (`probable`-tier) events, against 99 patients with a
usable time origin. This is explicitly below what a classical Cox model's EPV rule would recommend for more than
1-2 covariates — the reason the primary model uses hierarchical Bayesian shrinkage with informative priors rather
than an unpooled frequentist fit (see `model/approach.md` §2, §4). The pipeline still computes a dual-column label
schema (`event` / `event_sensitivity_incl_pending`) as a methodological safeguard, but with zero patients remaining
`pending_physician_reconfirmation`, both columns now coincide at 8 events — the primary/sensitivity split reported
in earlier drafts of this protocol no longer applies. See `model/approach.md` and
`reports/head_a_validation_report.md` for the final numbers.

### Train / Validation / Test Split
None used. At this event count, a hold-out split would leave single-digit events per arm — not a meaningful test.
Master Prompt Section 8 Level 3 specifies Harrell bootstrap optimism correction (500 resamples) instead; designed
but not yet executed in this build (`reports/head_a_validation_report.md` states exactly what has and hasn't been
run). All resampling/shuffling that *has* been run (the Level 4 leakage permutation test) operates at the whole-
patient level.

No oversampling or SMOTE is used anywhere, per Hard Rule #6 — the primary/expanded-endpoint imbalance (11 or 18
events vs. 99 patients) is handled by the Bayesian model's own likelihood and priors, not by resampling the data.

### Evaluation Metrics
> **Final, 2026-09-17, on the physician-confirmed 8-event cohort.** All bootstrap jobs referenced below have
> completed — see `reports/head_a_validation_report.md` Level 3 for the full table and discussion.

Uno's C-index (approximated here via `lifelines.utils.concordance_index` on a midpoint-time approximation, since
scikit-survival's IPCW C-index implementation could not be installed in this environment — see `model/approach.md`
§2), reported **Harrell bootstrap-corrected with 95% CI**, not as a naive in-sample point estimate (Master Prompt
§8 Level 3): penalized Cox 0.546 apparent → **0.536 corrected, 95% CI [0.467, 0.573]**; XGBoost AFT 0.507 →
**0.497, CI [0.436, 0.513]** (essentially exact chance); valve-family-only Weibull collapses to a degenerate 0.500
(213/500 = 43% of resamples fail to fit at all — zero events in some resampled family); literature-prior-only
(null, not bootstrapped — zero fitted parameters) apparent C-index is 0.493, essentially chance. **Primary
hierarchical Bayesian Weibull AFT: 0.618 apparent → 0.554 bootstrap-corrected, 95% CI [0.353, 0.771]** (B=100 full
4-chain MCMC per resample — the other three comparators used B=500; a stated ~10x compute-time tradeoff, not a
method difference, documented next to this result rather than in a footnote). **Plainly:** in about 55 of 100
random pairwise patient comparisons, the model correctly ranked who failed first — barely above chance on its
central estimate, with a 95% interval spanning worse-than-chance to fairly strong; "the model discriminates"
cannot be claimed with confidence at n=8, though its point estimate remains the best of the five models reported.
Posterior diagnostics for the primary model's one reported full-MCMC fit converged cleanly (R-hat = 1.00, 7
divergences/8000 draws ≈0.09%); the B=100 bootstrap resamples' own diagnostics were somewhat noisier than that
single fit (mean max R-hat 1.022, ~1.16% divergence rate), as expected for 100 unattended refits vs. one carefully
monitored fit. A case-level check tells the same story concretely: of the 8 confirmed-event patients, only 3 had
their actual event fall inside the model's own 80% credible interval for predicted median survival time — n=8 is
indicative only, not statistically robust, but is consistent with the Level 5 simulation study's own
under-coverage finding (48-56% observed vs. 80% nominal). Time-dependent AUC, integrated Brier score, and
calibration plots are designed but not yet computed.

### Subgroup Analyses
By implant approach (SAVR/TAVR/ViV-TAVR) and, where recoverable, valve family — both are structural components of
the hierarchical model itself (partial pooling), not a post-hoc subgroup cut. **Valve-family-level subgroup output
specifically is not a confirmed finding at this n** — the Level 5 simulation study found the model recovers
non-trivial family-pooling variance (`tau_family` ≈ 0.34) even with zero true simulated family heterogeneity,
matching the real fit's own 0.30, so family-level differences cannot currently be distinguished from that pooling
artifact (`model/approach.md` §5, `reports/head_a_validation_report.md` Level 5). Age-based subgrouping is not
possible (age unrecoverable, see `model/approach.md` §3).

### Comparator / Baseline
Four comparators, all reported alongside the primary model rather than in isolation (§5.4 requirement):
literature-prior-only (null), valve-family-only frequentist Weibull, penalized Cox (elastic net), XGBoost AFT. The
primary model's bootstrap-corrected C-index edge over all three data-driven comparators (0.554 vs. 0.536/0.497/
0.500 corrected — see Evaluation Metrics above for full CIs and the noted B=100-vs-B=500 resample-count difference,
and the family-only model is literally undefined for 2 of 3 observed families due to zero events) is presented as
evidence that literature-informed partial pooling is doing real work at this sample size — not as proof of
clinical utility, and not as a controlled head-to-head race given that asymmetry.

## 6. Ethical Considerations and Data Privacy

### IRB / Ethics Review
A production study on this design would require IRB review; retrospective analysis of already-de-identified data of
the kind provided for this hackathon typically qualifies for an expedited/exempt determination under most
institutional policies, but that determination was not sought or obtained for this prototype and should not be
assumed.

### Data Privacy
All source data was provided already de-identified (names, granular dates, addresses, ages replaced with
placeholder tokens). No re-identification attempt was made. See `data/data_plan.md` §6 for governance detail.

### Algorithmic Fairness
Not evaluable in this build: sex, race/ethnicity, and age are not recoverable from this de-identified extract at
all (age is masked with zero surviving values corpuswide), so no demographic-subgroup fairness analysis could be
performed. This is stated as an open gap, not glossed over — a production deployment with structured demographic
fields would need this analysis before go-live, particularly since the underlying literature priors are drawn from
cohorts with their own demographic skew (e.g. TAVR trial populations trending markedly older) that is currently
propagated into our priors unexamined.

### Clinical Transparency
Every model output carries the disclaimer that factors are statistical associations from n=99 patients/8 events, not
causal effects, and that the underlying severity labels are pipeline-derived with physician sign-off received on
all adjudicated disagreements (see `model/approach.md` §7 and `review/error_catalogue.md` §8) — sign-off covers the
label *values* used, not an independent clinical validation of the model's forecasts. The system is designed as a surveillance-scheduling decision-support flag — never a
standalone trigger for reintervention — and every Head B stage assignment carries its source sentence for a
clinician to verify directly rather than take on faith.
