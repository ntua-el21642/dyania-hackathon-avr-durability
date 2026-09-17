# Data Plan

---

## 1. Data Sources — Real Deployment

| Source | Data Type | Access Pathway | Variables Used |
|---|---|---|---|
| EHR clinical notes (operative reports, progress notes, procedures, discharge summaries) | Free text | HL7/FHIR DocumentReference feed or direct EHR export, structured-field-first parsing with narrative fallback | Implant event, reintervention/ViV, explicit SVD/BVF language, morphology, AR/gradient mentions, exclusion terms — everything Head B currently extracts |
| Structured echo reports | Semi-structured (template fields) | Cardiology imaging system export | Mean/peak gradient, EOA, DVI, AR grade, LVEF — currently only recoverable from free text in this extract; a production deployment with structured echo reports would materially improve coverage and precision over the current regex extraction |
| Labs (creatinine/eGFR, NT-proBNP, INR, etc.) | Structured, longitudinal | LIS/EHR export | Descriptive only in this build (see §4); a full-coverage deployment could promote these to real Head A covariates |
| Medications | Structured | Pharmacy/EHR export | Descriptive only in this build (same reason) |
| Vital status / death registry | Structured | State death index or EHR mortality flag | Not available in this extract at all — needed to model the competing risk of death, currently undocumented as BVF Stage 3 |

## 2. Data Sources — Prototype / Hackathon

| Dataset | Size Used | Why Selected | Limitations as Proxy |
|---|---|---|---|
| `notes_deidentified.xlsx` | 215 notes / 117 patients | Only dataset provided that covers the full cohort; richest source of implant, reintervention, and hemodynamic information despite being free text | Free-text extraction is inherently lossy vs. structured fields; year-only dates (see §4); heavy de-identification masking |
| `labs_deidentified.xlsx`, `medications_deidentified.xlsx` | ~43k lab rows / ~5.8k medication rows, but only 17/117 patients | Only patients with any lab/med data available in this extract | Not usable as SVD predictors here at all — see §4's "missing by design" note; used descriptively (cohort profile) only |
| Published literature (Capodanno et al. 2017, Trimaille et al. 2025, VARC-3/Généreux et al. 2021) | 3 PDFs, re-verified against source text on 2026-09-17 | Only way to inform a Bayesian prior with real published durability estimates at this sample size | Different eras, definitions, and populations than our cohort (e.g. TAVR cohorts skew markedly older); CV-inflated priors partially account for this, not eliminate it |

No synthetic or simulated data was used anywhere in this build.

## 3. Availability Assumptions

- **High-risk assumption, flagged:** we assume the 17 patients with labs/medications are not systematically
  different from the other 100 in ways that matter for SVD risk. We cannot test this (the missingness is
  block-wise by extraction, not by patient characteristic — see §4), and it is exactly the kind of selection bias a
  production deployment must check for before trusting any labs/meds-derived feature learned elsewhere.
- **Assumption that would NOT hold at most sites without work:** "valve model/family consistently coded" — only
  58/117 patients (50%) have a recoverable labelled valve size, and only 55/117 (47%) a recovered valve family name,
  purely from free-text pattern matching against a hand-built dictionary. A site with structured implant-device
  fields (e.g. UDI capture) would not have this gap.
- **Assumption that materially affects Head A:** "every implant event is described in an Operative Report or
  Procedures note this extract actually contains." 18/117 patients (15%) have no recoverable index-implant year at
  all and are excluded from the durability model entirely.

## 4. Preprocessing and Data Quality

### Data Cleaning

- Labs excluded as noise per Master Prompt Section 2.2: arterial blood gases, point-of-care glucose,
  respiratory-therapy entries, and the `TRANSCRIPTION` component (8,677 rows, no gradient content).
- **Line-wrap repair (found and fixed this session):** the source text conversion frequently splits a clinical
  phrase mid-word across a line break (e.g. `"...replacement of the aortic\nvalve."`) — every extractor in this
  pipeline matches literal-space phrases, so an unrepaired wrap silently makes an entire implant event or
  reintervention invisible. `sectioning.repair_line_wraps()` collapses exactly this pattern (a newline between a
  lowercase/`,`/`;` character and a following lowercase letter) before any extraction runs. Verified against the
  full 215-note corpus to only ever *lengthen* previously-truncated structured-field values, never change a
  negation-critical field. Full root-cause writeup: `review/error_catalogue.md` finding 3.1.
- Every hemodynamic value is attributed pre-implant (native valve) vs. post-implant (prosthetic) using the note's
  own Service Date relative to the patient's index implant year — **strictly after**, not on-or-after, since a note
  filed the same calendar year as the index operation is very often the pre-operative workup describing the native
  valve's own severe stenosis (observed directly in this corpus).

### Missing Data Strategy

- **No multiple imputation, no SMOTE/oversampling anywhere**, per Hard Rule #6 (Master Prompt §0) — the
  labs/medications block-wise gap (17/117 patients) and the valve-model/size gaps (~50% of the cohort) are both
  missing-by-extraction-design, not missing-at-random in a way a random-draw imputer should paper over.
- Missingness indicator columns (`valve_family_known`, `size_known`) are retained alongside every feature so a
  reader can distinguish "recovered and normal" from "not recoverable" — never silently merged.
- Patients with no recoverable index-implant year (18/117) are excluded from Head A rather than assigned an
  arbitrary time origin, and this exclusion count is reported in every run (`src/features/build_features.py`).

### Temporal Alignment

- **Interval censoring, not point estimates:** every date in this corpus is year-only. An event reported in year Y
  for a patient implanted in year Y₀ is treated as occurring somewhere in `[Y-Y₀-1, Y-Y₀+1]` years post-implant, per
  Master Prompt Hard Rule/Section 0 — implemented directly as the Head A likelihood's interval bounds, not
  approximated away except in the frequentist comparators, which use a documented midpoint approximation and are
  labelled as such everywhere they appear.
- Landmark = the index implant event itself (Operative Report/Procedures note, or — for 6/117 patients with no
  qualifying op report in this extract — the earliest dated narrative mention). There is no separate "30-day
  post-implant echo" landmark distinct from the index event in this notes-only extract; baseline echo is
  recoverable for only a minority of patients (135 gradient values across ~50 patients, and only one patient has
  gradients from ≥2 distinct years), so VARC-3-strict staging (which requires a baseline) is the exception, not the
  rule — the Capodanno absolute-threshold fallback (tagged `probable`, never `definite`) is what most `hvd_stage`
  assignments actually use.

### Label / Ground Truth Construction

Ground-truth hierarchy exactly as specified in Master Prompt Section 5.3 (definite reintervention / definite
explicit SVD text / probable hemodynamic threshold / possible morphology-or-trend-only / excluded competing event /
censored), implemented in `varc3_rules.py` and validated in two passes:

1. **Seed-set validation** against the physician's manually-reviewed 19-patient table (Master Prompt §2.4):
   reintervention-detector recall/precision/F1 went from 0.875/1.000/0.933 to **1.000/1.000/1.000** after fixing
   four root-caused extraction bugs found during this validation (full writeup: `review/error_catalogue.md`).
2. **Full-corpus re-run** after the fixes: confidence-tier distribution moved from
   censored 88 / definite 14 / probable 13 / possible 1 / excluded 1 to
   **censored 82 / definite 13 / probable 18 / possible 1 / excluded 3** (117 patients total; `bvf_stage==2` count
   11, up from 10) — every one of the 9 corpus-wide label changes individually inspected against raw note text
   before accepting the run, including catching and re-fixing a false positive the fix itself introduced
   (Patient_034) before it reached `labels.csv`.

**Full 117-patient physician blind adjudication is complete, and sign-off has been received on every disagreement**
(`review/adjudication_form.xlsx`, sign-off 2026-09-17). It disagreed with the pipeline on 11/117 patients.
Mechanical re-verification against raw note text, one code fix (valve-context disambiguation, Patient_071), and one
confirmed correction (Patient_077, a bare coded-history false positive) resolved 7 of the 11 directly. The
remaining 4 (058, 061, 068, 081) needed a physician judgment call and were held as
`pending_physician_reconfirmation=True` until sign-off; resolution was **asymmetric, not a blanket acceptance of
either side** — 058, 061, and 068 confirmed the pipeline's original evidence-based read (no label change), while
081 confirmed the physician's own read over the pipeline's (a genuine pipeline over-call, documented as a known,
not-yet-fixed extractor gap in `config/label_overrides.yaml`). **Final prevalence:** 8/117 (6.8%) physician-confirmed
BVF Stage 2 (hard endpoint, all `confidence_tier=definite`); zero patients remain `pending_physician_reconfirmation`;
22/117 (18.8%) `probable`-tier HVD signal (expanded/secondary endpoint), unaffected by this round. Full audit trail:
`review/error_catalogue.md` §8.

## 5. Synthetic or Proxy Data

None used. The only proxy in this build is the PPM indicator (labelled valve size ≤21mm, standing in for true
iEOA, which requires BSA that isn't recoverable here) — documented as a proxy everywhere it appears, per Master
Prompt Section 5.1 #8.

## 6. Data Governance and Privacy

All data provided is already de-identified (names, dates beyond year, addresses, ages replaced with placeholder
tokens such as `[NAME]`, `[DATE]`, `[AGE]`). No re-identification attempt was made at any point. This repository
(per its own README template) is treated as potentially publicly visible, so only de-identified derivatives
(labels, features, aggregate statistics, posterior summaries) are committed — never raw free-text notes beyond what
was already provided de-identified in the source files themselves. A production deployment would require a
site-specific DUA, IRB determination (see `protocol/study_protocol.md` §6), and a formal PHI handling/audit plan
that this hackathon prototype does not need to implement.
