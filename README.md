# Dyania Health Hackathon 2026 — Team Submission Repository

**Challenge:** Build a study — using machine learning, a statistical model, or whatever approach you prefer — proposing a protocol to predict aortic valve durability in patients with a bioprosthetic aortic valve replacement.
**Event:** September 15–17, 2026 (3 days)

---

## Problem Statement

Bioprosthetic aortic valves wear out. Structural valve deterioration (SVD) is currently caught by fixed-interval
echocardiographic surveillance — a strategy that misses the individual variation in when a given valve, patient, or
device family actually starts failing. By the time a patient presents symptomatically with severe hemodynamic
deterioration, the reintervention (redo surgery or valve-in-valve TAVR) is often performed under worse conditions
than an earlier, risk-stratified referral would have allowed. In our own 117-patient cohort, **8 reinterventions
were identified and physician-confirmed** (all SAVR-index — see below); several patients (e.g. one case with an
emergent, cardiogenic-shock valve-in-valve) illustrate the cost of late identification directly. A system that
stages current SVD severity from
routinely collected notes and forecasts individual durability could shift surveillance from fixed-interval to
risk-adaptive.

## Our Approach

We built a two-headed, notes-first pipeline on 117 de-identified patients (215 notes; only 17 of the 117 also have
labs/medications — see [Key Design Decisions](#key-design-decisions)):

- **Head B (severity, deterministic):** a rule-based NLP engine — sectioning, negation/template handling, temporal
  attribution, valve-dictionary extraction — implements VARC-3 (Généreux et al. 2021) as primary and Capodanno/EAPCI
  2017 as a documented sensitivity/fallback, producing an auditable HVD stage (0–3), BVF stage (0/2/3), S/R/RS
  phenotype, and a definite/probable/possible/excluded/censored confidence tier for every patient, with every
  assignment traceable to a source sentence.
- **Head A (durability, probabilistic):** a hierarchical Bayesian Weibull AFT (PyMC) with an interval-censored
  likelihood (year-only dates mean every event time is only known to ±1 year), literature-informed priors fit from
  three verified published sources, and partial pooling of valve family toward approach (SAVR/TAVR), reported
  alongside four comparators (literature-prior-only, valve-family-only frequentist Weibull, penalized Cox, XGBoost
  AFT).

We chose this over an end-to-end black-box model because n=117 with only 8 confirmed events cannot responsibly
support anything more complex, and because a physician reviewer needs to see and correct every label before any
model trained on it is trusted — see [`review/error_catalogue.md`](review/error_catalogue.md) for a full account of
validating and fixing the NLP pipeline: four root-caused bugs found and fixed against a physician-reviewed 19-patient
seed set (reintervention-detector F1 improved from 0.933 to 1.000), followed by a full 117-patient blind physician
adjudication that surfaced a much lower blind-read F1 (0.353, near-chance kappa) and 11 disagreements, all of which
are now resolved with physician sign-off received 2026-09-17.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Two separate heads (severity now, durability forecast) rather than one endpoint | VARC-3 defines BVF as the endpoint of a staged process (HVD 1→2→3→BVF); a single binary label would throw away the staging information that is the whole clinical point, and the two heads have genuinely different evidence requirements (deterministic rules vs. probabilistic time-to-event). |
| Deterministic rule engine for severity, not a trained classifier | At n=117 with sparse baseline echo, a physician must be able to trace every label to its source sentence for Level-1 validation (Section 8) — this also directly produces the weak-supervision labels Head A trains on, so its own precision has to be defensible first (see error_catalogue.md). |
| Hierarchical Bayesian AFT with literature priors, not a purely data-driven survival model | 8 events cannot identify a rich model on their own. Partial pooling + literature-anchored priors (verified against the actual source PDFs, not taken on faith — see `config/priors.yaml`) let the model borrow strength honestly, and the posterior vs. prior comparison is itself a finding worth reporting. |
| 11/117 disagreements from full-cohort blind physician adjudication, all resolved before final labels | Mechanical raw-text re-verification and one code fix resolved 7 directly (`review/error_catalogue.md` §8); the remaining 4 (058, 061, 068, 081) needed a physician judgment call and were held as `pending_physician_reconfirmation` until sign-off — received 2026-09-17, asymmetrically (058/061/068 confirmed the pipeline; 081 confirmed the physician, over the pipeline's over-call). Final confirmed count: 8/117 events. See `config/label_overrides.yaml`. |
| Age excluded as a covariate; "age-only" comparator not built | `[AGE]` is masked in every note in this corpus, with zero surviving numeric values anywhere (verified by a full-corpus regex scan). We report this as a limitation rather than fabricating or imputing an age. |
| Labs/medications used descriptively only, never as model features | None of the 8 confirmed SVD reinterventions fall in the 17-patient labs/meds subset (the two reinterventions that subset does contain are both non-SVD: endocarditis and probable PVL) — training a "labs+meds" predictive model on zero SVD events would be fabricating signal. Documented as missing-by-design (block-wise extraction gap), with an explicit note that this may not generalize to a deployment setting where labs/meds exist for every patient. |
| Physician adjudication ran in parallel with, not gating, modeling — then closed the loop before submission | Given the 3-day timeline, `review/adjudication_form.xlsx` covered all 117 patients while Head A/B development proceeded on the pipeline's own (validated, bug-fixed) labels; sign-off on the 4 outstanding disagreements was received 2026-09-17 and Head A was re-run on the final label set before this submission. |

---

## Repository Structure

```
.
├── README.md                        # This file
├── protocol/study_protocol.md       # Full study design
├── model/approach.md                # Model methodology and validation strategy
├── data/data_plan.md                # Data sources, preprocessing, availability
├── presentation/slides.md           # Slide outline
├── evaluation/scoring_rubric.md     # Hackathon rubric (reference)
├── review/
│   ├── error_catalogue.md           # Phase 2 pipeline validation: seed-set P/R/F1, 4 root-caused
│   │                                 # bugs found+fixed, before/after corpus-wide impact
│   └── adjudication_form.xlsx       # Blind physician review form, all 117 patients (Patient_061
│                                     # pre-filled with a suggested/justified determination)
├── config/priors.yaml               # Literature-derived Weibull priors, verified against source PDFs
├── reports/                         # Head A posterior, comparator outputs, forest/SHAP plots,
│                                     # head_a_validation_report.md
├── tests/test_varc3_rules.py        # Head B unit tests (one case per VARC-3 stage + edge cases)
├── notes_deidentified.xlsx / labs_deidentified.xlsx / medications_deidentified.xlsx   # raw data
├── labels.csv, notes_audit.csv      # Head B output (117 patients, note-level audit trail)
├── data_processed_patient_features.csv / data_processed_patient_labels.csv            # Head A input
├── sectioning.py, negation.py, temporal.py, valve_dictionary.py,
│   extract_core.py, varc3_rules.py, pipeline.py, reviewer_sheet.py   # Head B pipeline
├── src/
│   ├── features/build_features.py            # Landmark feature + interval-censored label construction
│   ├── models/durability/
│   │   ├── fit_literature_priors.py           # Section 6.2 prior fitting
│   │   ├── weibull_aft_bayes.py                # Primary Head A model
│   │   └── comparators.py                      # Null/family-only/Cox/XGBoost AFT comparators
│   ├── explain/durability_explain.py           # Forest plot + SHAP
│   └── validation/leakage_test.py              # Section 8 Level 4
└── notebooks/                       # Mirrored copy of the pipeline for standalone execution
```

## Reproducing this submission

```
pip install -r requirements.txt
pip install pymc arviz lifelines xgboost shap pyyaml h5netcdf h5py   # Head A stack (see model/approach.md
                                                                      # for the scikit-survival substitution note)
python pipeline.py                              # Head B: notes -> labels.csv, notes_audit.csv
python -m pytest tests/                          # Head B unit tests
python src/models/durability/fit_literature_priors.py   # -> config/priors.yaml
python src/features/build_features.py            # -> data_processed_patient_*.csv
python src/models/durability/weibull_aft_bayes.py        # Primary Head A model (~30-40s on this cohort size)
python src/models/durability/comparators.py
python src/validation/leakage_test.py
python src/explain/durability_explain.py
python reviewer_sheet.py                          # -> review/adjudication_form.xlsx
```

All scripts use fixed seeds (20260917). On Windows without a C++ build toolchain, PyMC's sampling requires
`PYTENSOR_FLAGS="cxx="` (pure-Python fallback — see `model/approach.md`).

## Status — what's final and what's genuinely still open

- **Physician adjudication of all 117 patients is complete, and sign-off has been received on every disagreement**
  (`review/adjudication_form.xlsx`, sign-off 2026-09-17). Full-cohort blind review disagreed with the pipeline on
  11 patients; 7 resolved via mechanical raw-text re-verification/code fixes, the remaining 4 (058, 061, 068, 081)
  via physician sign-off (asymmetric: 058/061/068 confirmed the pipeline, 081 confirmed the physician). **Final
  confirmed event count: 8/117** — zero patients remain `pending_physician_reconfirmation`. Full account:
  `review/error_catalogue.md` §8, `config/label_overrides.yaml`.
- Validation Levels 1 (label validity), 2 (construct/known-groups), 3 (bootstrap-corrected C-index), 4 (leakage),
  and 5 (simulation study) are all re-run on this final 8-event cohort — see `reports/head_a_validation_report.md`
  for the current numbers. VARC-3-vs-Capodanno and prior-informativeness sensitivity checks remain not yet run.
- **Genuinely still open:** a second physician rater for inter-rater kappa (only one physician's adjudication is
  reflected in the current labels); Guyot KM-curve reconstruction (Master Prompt Section 5.6) — algorithm
  implemented and self-tested on synthetic data (`src/external/guyot.py`), and a concrete digitization checklist
  exists (`data/external/km_digitized/DIGITIZATION_CHECKLIST.md`) — but never run on real data, since none of the
  3 provided source PDFs contain an actual KM curve with a numbers-at-risk table (confirmed by full-text search);
  and the documented `extract_implicit_new_tavr` corroborating-finding gap (`config/label_overrides.yaml`,
  Patient_081) is specified but not yet fixed in code.
