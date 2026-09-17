# Presentation Slide Content

> Source content for `slides.pdf` (5–10 slides, export from this outline). Each slide below is the actual
> bullet content, not a generic prompt — pull numbers directly from `review/error_catalogue.md`,
> `reports/head_a_validation_report.md`, and `config/priors.yaml`.

---

## Slide 1 — Title

- [Team name] — Dyania Health Hackathon 2026
- Notes-Derived SVD Staging and Bayesian Durability Forecasting for Bioprosthetic Aortic Valves
- 2026-09-17

---

## Slide 2 — The Clinical Problem

- Structural valve deterioration (SVD) is currently caught by fixed-interval echo surveillance, not individualized
  risk — every patient gets the same schedule regardless of their actual trajectory.
- In our own 117-patient cohort: **8 physician-confirmed reinterventions**, all SAVR-index, spanning routine ViV
  TAVR to one documented emergent case (cardiogenic shock at presentation) — a direct illustration of what late
  identification costs.
- Published durability ranges from ~92% freedom-from-SVD at 5 years (Mitroflow) to ~96% at 10 years (Freestyle) —
  device choice alone spans a wide durability range surveillance schedules don't currently account for.

## Slide 3 — Current Workflow Failure

- Fixed-interval echo surveillance treats a Freestyle and a Mitroflow patient identically, despite materially
  different published degradation curves.
- Reintervention indication is often only captured retrospectively in free text — no structured field in this
  corpus flags "SVD" directly; extracting it required a purpose-built NLP pipeline (below).
- Bottleneck: valve model/size is only recoverable from free text for about half our cohort (58/117), and labs/
  medications exist for only 17/117 patients in this extract — real-world data completeness is itself part of the
  problem.

## Slide 4 — Our Hypothesis

- If durability risk were forecast from routinely collected notes at implant time, surveillance could shift from
  fixed-interval to risk-adaptive — closer follow-up for short-predicted-durability patients, standard intervals for
  the rest.
- At n=99/8 events, we can't yet validate a deployable risk score — but we can validate the *method*: an
  auditable severity-staging engine plus a literature-anchored Bayesian durability model that degrades honestly
  (wide uncertainty) rather than overclaiming at small n.

---

## Slide 5 — Study Overview

- Population: bioprosthetic AVR recipients (SAVR/TAVR/ViV) with a recoverable implant year; 117 patients (Head B),
  99 with a usable time origin (Head A).
- Primary endpoint: BVF Stage ≥2 (reintervention), VARC-3-staged — **8 physician-confirmed events**, final
  (`review/error_catalogue.md` §8). Secondary: moderate-or-greater HVD (22/117 patients, `probable`-tier).
- Ground truth: 5-tier hierarchy (definite/probable/possible/excluded/censored), validated in two passes — a
  19-patient seed set (reintervention-detector **F1 = 1.000** after fixing 4 root-caused pipeline bugs, from
  0.933 pre-fix), then a full 117-patient blind adjudication with physician sign-off received on all 11
  disagreements it surfaced.

## Slide 6 — Data Sources

- Prototype: 215 de-identified notes / 117 patients (only source covering the full cohort); labs/medications for
  17/117 only, used descriptively — **zero of our 8 confirmed SVD reinterventions fall in that 17-patient subset**,
  so we do not train any predictive feature on labs/meds (would be fabricating signal from zero events).
- Real deployment: structured echo reports and implant device (UDI) fields would close the ~50% valve-model/size
  gap this free-text extraction currently has.
- Age is masked with **zero surviving numeric values anywhere in the corpus** — flagged as a hard limitation, not
  worked around.

## Slide 7 — Validation Plan

- No train/test split at 8 events — Bayesian partial pooling + literature-informed priors instead of a
  classical split; leakage controls (feature-timestamp check, temporal ordering check, label-permutation test)
  all pass.
- Metrics: posterior diagnostics (R-hat = 1.00, clean convergence) + Harrell bootstrap-corrected C-index (95% CI)
  against 4 comparators.
- Comparator baseline: literature-prior-only, valve-family-only frequentist Weibull (undefined for 2/3 families —
  zero events), penalized Cox, XGBoost AFT.

---

## Slide 8 — Model Architecture

- Two heads: deterministic VARC-3 rule engine (severity, fully auditable — every label traces to a source
  sentence) + hierarchical Bayesian Weibull AFT with an interval-censored likelihood (durability).
- Interval censoring is load-bearing, not cosmetic: every date in this corpus is year-only, so every event time is
  only known to ±1 year — implemented directly in the model's likelihood, not approximated away.
- Priors fit from 3 published sources, re-verified against the actual source PDFs (not taken on faith); Trifecta
  has no literature durability point in any of the 3 papers (confirmed by full-text search) and is deliberately
  given no fabricated prior.

## Slide 9 — Results

> Final numbers below are from the physician-confirmed 8-event cohort — see `reports/head_a_validation_report.md`
> Level 3 for the full table and CIs.

- Bootstrap-corrected C-index (Harrell optimism correction, 95% CI — not a naive point estimate): primary Bayesian
  model **0.618 apparent → 0.554 corrected, 95% CI [0.353, 0.771]** vs. penalized Cox 0.536 CI [0.467, 0.573],
  XGBoost AFT 0.497 CI [0.436, 0.513], literature-only 0.493 (essentially chance, not bootstrapped). Plainly: in
  about **55 of 100 random pairwise patient comparisons**, the model correctly ranked who failed first.
- Honest read, shown explicitly, not hidden: with only 8 confirmed events, these intervals are wide — the primary
  model's own 95% CI spans worse-than-chance to fairly strong. The literature-only model scoring at chance on our
  own cohort is itself a key finding: it's why a Bayesian update, not literature transfer alone, is the right
  framing. A stated compute-time asymmetry (B=100 for the primary Bayesian model vs. B=500 for the frequentist
  comparators) is flagged directly next to the results table, not buried in a footnote.
- Case-level check (n=8, indicative only): of the 8 confirmed-event patients, **3 had their actual event fall
  inside the model's own 80% credible interval** for predicted median survival time — consistent with the Level 5
  simulation study's own under-coverage finding (48-56% observed vs. 80% nominal), not a new surprise.
- Reintervention detector (Head B), final after full physician reconciliation: **F1 = 0.889** (precision 0.800,
  recall 1.000 — the automated detector found all 8 true reinterventions, with 2 false positives caught and
  corrected by physician review).
- SHAP/forest-plot outputs available (`reports/head_a_forest_plot.png`, `head_a_shap_summary.png`), reported with
  an explicit `tau_family` pooling-artifact caveat (Level 5 simulation study) on any family-level claim.

---

## Slide 10 — Impact and Next Steps

- What this enables today: a validated, auditable NLP severity pipeline (F1=1.000 on seed set) that a physician
  reviewed and signed off on end-to-end, ready to scale to a larger extract without re-architecting.
- Full validation in a real health system would still require: a second physician rater for inter-rater kappa
  (only one physician's adjudication is reflected in the current labels), and real Guyot KM-curve digitization
  (algorithm implemented + self-tested, but none of our 3 source papers contain an actual curve to digitize — need
  the primary studies, see `data/external/km_digitized/DIGITIZATION_CHECKLIST.md`). All five validation levels
  (label validity, construct validity, bootstrap, leakage, simulation) are complete on the final 8-event cohort —
  see `reports/head_a_validation_report.md`.
- 3-month next step: multi-site extract to push past single-digit confirmed events, structured echo integration to close the
  valve-model/size gap, a formal fairness analysis once demographic fields are available, and fixing the documented
  `extract_implicit_new_tavr` corroborating-finding gap (`config/label_overrides.yaml`, Patient_081).
- Deployment readiness: method-validated, not clinically validated — the honest next milestone is n in the low
  hundreds of confirmed events, not a production rollout.
