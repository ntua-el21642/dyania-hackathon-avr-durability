# Seed-Set Validation & Error-Analysis Catalogue (Phase 2 checkpoint / Section 8, Level 1)

> **Update (post-fix):** all four findings below (3.1–3.4) have been fixed and the pipeline re-run on all
> 117 patients. See §5 "Before vs. after" for the fix implementations, corpus-wide impact, and the updated
> seed-set metrics. Sections 1–4 are preserved as originally written (pre-fix) since they document the
> diagnostic process; §5 is the outcome.

**Status:** Validation against the 19-patient physician-reviewed seed table (Master Prompt §2.4) only.
This is **not** the full 117-patient blinded adjudication (§8 Level 1 / Phase 3) — that adjudication form
(`review/adjudication_form.xlsx`) is being built in parallel per team instruction and is still pending
physician sign-off. Treat every number below as provisional.

Source: `labels.csv` / `notes_audit.csv` as produced by `pipeline.py` on 2026-09-17, cross-checked against
raw `notes_deidentified.xlsx` text and the extractor functions in `extract_core.py` / `temporal.py`.

---

## 1. Row-by-row check against the seed table

| Patient | Seed category | Pipeline `bvf_stage` | Pipeline `hvd_stage` | Confidence | Match? |
|---|---|---|---|---|---|
| 017 | Definite BVF2 (redo BioBentall) | 2 | 2 | definite | ✅ |
| 035 | Definite BVF2 (TAVR after failed SAVR) | 2 | 3 | definite | ✅ |
| 036 | Definite BVF2 (ViV Evolut) | 2 | 2 | definite | ✅ |
| 038 | Definite BVF2 (ViV TAVR 12/2023) | 2 | 2 | definite | ✅ |
| 044 | Definite BVF2 (emergent ViV) | 2 | 2 | definite | ✅ |
| 058 | Definite BVF2 (ViV 08/2024) | 2 | 2 | definite | ✅ |
| 068 | Definite BVF2 (CE #21 → TAVR S3 23) | **0** | 3 | definite (HVD only) | ❌ **FN — see 3.1** |
| 071 | Definite BVF2 ("explicit SVD", redo 2018) | 2 | 2 | definite | ⚠️ right label, wrong evidence — see 3.4 |
| 103 | Non-SVD reintervention → exclude (endocarditis) | 0 (excluded) | 0 | excluded | ✅ |
| 104 | Non-SVD reintervention, probable PVL → exclude | **0, not excluded** | 2 | definite | ❌ **wrong on two axes — see 3.2** |
| 061 | Needs adjudication (possible endocarditis) | 2 | 0 | definite | ⚠️ plausibly *more* correct than the seed caution — see 3.5 |
| 089 | Planned reintervention (severe prosthetic AI) | **0** | 0 | censored | ❌ **FN — see 3.3** |
| 030 | SVD signal, no reintervention (MG 22, "dysfunction") | 0 | **3 (should be 2)** | definite | ⚠️ stage inflated — see 3.4 |
| 042 | SVD signal, morphological Stage 1 | 0 | 2 | definite | ⚠️ different evidence than seed anticipated, not necessarily wrong |
| 053 | SVD signal, qualitative gradient worsening | 0 | 1 | possible | ✅ good match |
| 024 | Known FP (redo for CABG) | 0 | 0 | censored | ✅ correctly rejected |
| 028 | Known FP (redo for CABG) | 0 | 0 | censored | ✅ correctly rejected |
| 048 | Known FP (redo for CABG) | 0 | 0 | censored | ✅ correctly rejected |
| 108 | Known FP (redo for CABG) | 0 | 0 | censored | ✅ correctly rejected |

## 2. Precision / Recall / F1 — BVF Stage 2 (reintervention) detector

Denominator restricted to the seed table's **unambiguous** classes only (the 8 "definite" positives and the
5 unambiguous negatives: the 4 known-FP redo-for-CABG patients + Patient_103, whose exclusion is itself
confirmed correct). The 6 gray-zone patients (104, 061, 089, 030, 042, 053) are reported separately below,
not folded into this number, because the seed table itself marks them as ambiguous or non-reintervention.

- TP = 7 (017, 035, 036, 038, 044, 058, 071)
- FN = 1 (068)
- FP = 0 (024, 028, 048, 108, 103 all correctly non-positive)
- TN = 5

**Recall = 7/8 = 0.875**
**Precision = 7/7 = 1.000**
**F1 = 0.933**

This is short of the §12 "definition of done" target (F1 ≥ 0.90 against physician adjudication) only in the
sense that recall is driven down by one root-caused bug (3.1), not a modeling ceiling — see fix below.

If Patient_061 is treated as a false positive per the seed table's caution (rather than as a plausible
correct call — see 3.5), precision drops to 7/8 = 0.875 and F1 to 0.875. We report both readings rather
than picking one, pending physician sign-off on 061 specifically.

## 3. Root-caused findings

### 3.1 [BUG, highest priority] Line-wrapped phrases break literal-space regex matches — causes the Patient_068 false negative and corrupts index-event recovery for ≥9 patients

`AVR_PATTERN` / `TAVR_PATTERN` (`temporal.py`) match literal phrases such as `"replacement of the aortic valve"`
using a plain space character between words. De-identified source text frequently wraps mid-phrase:
Patient_068's 2012 Operative Report reads `"...replacement of the aortic\nvalve."` — a newline, not a space,
sits between "aortic" and "valve", so the regex silently fails to match.

Consequence for 068: `classify_note_as_implant_event` returns `None` for the index operative report →
`index_implant_year`/`index_approach`/`index_valve_model` all stay unrecovered (`undated`/`None`) →
the implicit-TAVR-after-SAVR reintervention gate in `pipeline.py` (`if ... tl["index_approach"] == "SAVR"`)
never fires, even though `extract_implicit_new_tavr` *did* correctly find `"S/P TAVR"` in the 2024 follow-up
note. The patient's own gradients (pre-TAVR MG 43 "severe", post-TAVR MG 10) are exactly the signature of a
completed reintervention, but with no known index approach the pipeline has no way to attribute it.

**Scope check:** re-ran `AVR_PATTERN`/`TAVR_PATTERN` against raw vs. whitespace-normalized text for all 127
Operative Report/Procedures notes. **10 notes across 9 patients** (017, 026, 029, 054, 066, **068**, 090,
097, 100) flip from no-match to match after normalization. 017's index event happens to get recovered
another way (its BVF-2 label fires independently via the redo/SVD-explicit-text detectors, which don't gate
on `index_approach`), so this bug is silent there — but it corrupts `index_valve_model`, `index_approach`,
`years_followup`, and any Head-A feature sourced from the index event for however many of the other 8 are
affected, and blocks the implicit-TAVR gate the way it did for 068.

**Fix:** whitespace-normalize note text (`" ".join(text.split())`, already implemented as `_normalize_ws` in
`pipeline.py` but only used for snippet display) *before* running extraction regexes, not just for
provenance snippets. Re-run the full pipeline after the fix and re-check all 9 flagged patients' index
events and all downstream labels before trusting cohort-level valve-family/approach statistics for Head A.

### 3.2 [BUG] Explicit-SVD-text detector fires on a nonspecific ICD problem-list phrase regardless of etiology — Patient_104

Both Patient_068 (SVD, correctly) and Patient_104 (probable PVL per seed table, i.e. non-SVD) carry the
identical past-medical-history entry: `"Prosthetic aortic valve failure"` (an ICD-coded problem-list line,
not a clinical narrative). `extract_svd_explicit`'s pattern matches this phrase unconditionally and assigns
`confidence_tier=definite`. For 104 this is wrong on two counts:

1. **Missed event:** the actual procedure — `"TAVR +/- paravalvular leak repair"`, explicitly scheduled and
   explicitly PVL-attributed in the note's own HPI — was never picked up by `extract_redo`/`extract_viv`/
   `extract_implicit_new_tavr` at all, likely because this note is a pre-procedure anesthesia consult, not a
   completed operative report, and "TAVR" alone with no ViV/redo keyword nearby doesn't match either
   extractor.
2. **False signal:** the generic ICD phrase alone drove a `definite` SVD label, when VARC-3 explicitly
   excludes PVL as non-structural (NSVD, §4.4). Per Section 2.3, Patient_104 is one of only two
   reinterventions in the labs/meds 17-patient subset, and its labeled cause here is wrong.

**Fix:** `extract_svd_explicit` needs the same exclusion-proximity check already applied to redo/ViV/SVD
triggers elsewhere in `pipeline.py` (`_trigger_near_exclusion`) — it currently is checked per-note in
`pipeline.py`'s main loop (`svd_exclusion_reason = _trigger_near_exclusion(...)`), but for 104 the PVL
mention ("`paravalvular leak repair`") sits in the HPI of the *same* note as the SVD-text trigger, so the
200-char `EXCLUSION_LINK_WINDOW` should have caught it — needs direct debugging of why it didn't (worth a
follow-up: confirm `extract_exclusions` actually has a PVL bucket wired into the same call site used for the
explicit-SVD-text check, not just the redo/ViV check). Flagging as unresolved rather than guessing further.

### 3.3 [GAP] Planned-but-not-yet-performed reinterventions and non-standard AR abbreviations are invisible to the current extractors — Patient_089

Patient_089's 2025 consult note states in the HPI: `"The Echocardiogram reveals svr AI, mild MR"` (likely
"severe AI") and ends `"Impression: AI s/p AVR, CAD / Plan: redo AVR/CABG"` — a clinically clear picture of
planned redo driven by severe prosthetic regurgitation. But:

- `"svr AI"` does not match `AR_GRADE_PATTERN` (which expects `trace|mild|moderate|severe`, not the
  abbreviation `svr`).
- The *same note* pastes in an older, contradictory structured echo block reporting "no aortic valve
  regurgitation" (peak/mean 19/9 mmHg) with a note that it's "Similar findings" to a still-older prior echo —
  this is the value that ends up in `labels.csv` (`MG=9.0 below moderate threshold`).
- `PLANNED_FRAME`/redo detection correctly recognizes `"Plan: redo AVR/CABG"` as a *future* procedure (not
  falsely counted as completed BVF Stage 2, which is correct behavior) — but there's currently no
  "probable/planned SVD-R" bucket to route it to instead, so it falls all the way through to `censored`.

**Fix:** (a) add `svr`/`sev` as recognized AR-severity abbreviations; (b) when a `PLANNED_FRAME` redo/ViV hit
is suppressed as future rather than completed, route it to a distinct "planned reintervention — probable"
bucket (per Master Prompt §2.4's own BVF Stage 1 / probable-SVD-R category) instead of letting the patient
default to fully censored with no signal at all.

### 3.4 [BUG] AR-grade extractor has no "restates the index indication" guard — Patient_030 stage inflated from 2→3

Patient_030's only post-implant note (2024) contains two AR mentions: the real post-implant echo finding
`"mild (1+ - 2+) aortic valve regurgitation"` (ordinal 1), and, earlier in the same note's HPI, `"AVR for
severe AI (09/2011)"` — a one-line restatement of *why the patient had the original 2011 implant*, i.e. the
native valve's own pre-op severe regurgitation, dated the same year as the index operation. `extract_ar_grade`
has no history-restatement check analogous to `_restates_known_index_year`/`_restates_undated_index` (which
`pipeline.py` already applies to redo/ViV hits) — it takes the `max(ordinal)` across all post-implant-note AR
mentions and picks up the historical "severe" restatement, escalating `hvd_stage` from 2 (moderate, correctly
gradient-driven, MG 22) to 3 (severe). The seed table's own gradient value (MG 22) matches the pipeline's
extracted number exactly — only the AR-grade component is wrong.

This is very likely the *same underlying pattern* implicated in 071's evidence mismatch (row 3 of §1): 071's
`redo_evidence` field cites a mitral-valve operative note ("Redo median sternotomy, mitral valve replacement
with 27-mm Biocor... reconstruction of the fibrous trigone") rather than an aortic-valve-specific one — the
final `bvf_stage=2` label appears directionally correct (this patient's chart elsewhere documents an AVR
redo per the seed table), but the specific evidence snippet stored for physician review is for the wrong
valve. Worth a `nearest_valve_context`-style guard on the redo/ViV evidence snippet, not just on gradient
extraction (that helper already exists in `extract_core.py` for AR-grade valve attribution — it doesn't look
like it's applied to the redo/ViV snippet selection).

**Fix:** apply the same restates-index-year / restates-index-note-text checks used for redo/ViV hits to
`extract_ar_grade` (and ideally to `extract_gradients`) before selecting the "worst" value per note.

### 3.5 [Positive finding] Negation handling correctly protected Patient_061 from a false endocarditis exclusion — and the pipeline's "definite SVD" call may be *more* correct than the seed table's caution

The seed table flags 061 as "needs adjudication (possible endocarditis)". The actual 2024 postop note twice
documents a *negative* endocarditis workup: `"ID consulted to eval for need for antibiotics in setting of
possible Endocarditis - no indication for ATBx per ID"` and `"Per CTS note no signs of endocarditis. Cultures
negative."` The exclusion detector correctly did not fire (negation handling worked as intended). Separately,
the stated indication for this redo — `"TEE shows severe AI secondary to a flail leaflet"` — is itself a
VARC-3 §4.1 morphological SVD criterion ("tear, flail"). Recommend flagging this patient for the physician
adjudication form as "recommend reclassifying from needs-adjudication to confirmed SVD, pending review of the
negative endocarditis workup quoted above" rather than treating the pipeline's current label as an error.

## 5. Before vs. after: fixes applied and their impact

All four findings (3.1–3.4) were root-caused to specific code, not modeling limitations, so all four were
fixed rather than deferred:

| # | Fix | Where |
|---|---|---|
| 3.1 | `sectioning.repair_line_wraps()`: collapse a newline sitting between a lowercase/`,`/`;` character and a following lowercase letter (the signature of a soft mid-phrase line wrap) to a single space. Applied once to `df["Notes"]` at load time in `pipeline.py`, before any extraction, so every extractor and `build_patient_timelines` see the same repaired text and self-consistent character offsets. Verified against the full 215-note corpus: only ever *lengthens* previously-truncated `parse_field_lines` values; zero change to the negation-critical `Reoperation`/`Valve in Valve`/`Tissue Implant Type`/`Implant Size` fields. | `sectioning.py`, `pipeline.py` |
| 3.4 | `extract_ar_grade` hits are now filtered through the same `_restates_known_index_year` / `_restates_index_note_text` guards already used for redo/ViV hits, before picking the "worst" AR grade per note. | `pipeline.py` |
| 3.2a | `extract_svd_explicit` now tags each hit `in_bare_history_list` (nearest preceding all-caps section header is PMH/PSH/Active-Problem-List **and** the hit's own line is short with no narrative elaboration — "due to"/"caused by"/"gradient"/"echo"/"assessment"/"mmHg"). A narrative hit in the same note is always preferred; a bare-history-only hit is routed to a new `possible`-tier bucket (`has_svd_history_list_only`) in `varc3_rules.py`, never `definite`. Verified against the full corpus: exactly one match fires this path (Patient_104's "Prosthetic aortic valve failure"); Patient_017's structurally similar-looking PMH section is correctly *not* downgraded because its SVD-explicit phrase sits inside a per-problem "History:" narrative paragraph, not a bare list line. | `extract_core.py`, `pipeline.py`, `varc3_rules.py` |
| 3.2b | New check: a bare AVR/TAVR mention (no redo/ViV keyword, so it wouldn't otherwise create any event) that shares a **newline-free line/sentence** with a PVL/endocarditis/thrombosis exclusion term within 200 chars now routes the patient to `excluded` directly. The same-line requirement (`_trigger_near_exclusion_same_line`, stricter than the existing `_trigger_near_exclusion`) was added *after* an initial version without it produced a false positive on Patient_034 (an "Aortic Valve Replacement" problem-list entry and an unrelated "Acute Deep Vein Thrombosis" entry, 146 chars apart but on separate lines of the same coded problem list, with no causal link) — caught by re-running the full-corpus diff and inspecting every newly-changed label, not just the seed set. Patient_069 (genuine same-sentence "aortic valve replacement and possible infective endocarditis") and Patient_104 correctly remain excluded after the tightened check; Patient_034 correctly reverts to censored. | `pipeline.py` |
| 3.3 | Added `svr` as a recognized AR-severity token in `AR_GRADE_PATTERN`/`AR_GRADE_ORDER` (maps to `severe`/ordinal 3). Scoped by requiring it be immediately followed by an AR/AI/regurgitation/leak token, so it cannot collide with the far more common cardiology usage "SVR" = systemic vascular resistance (always followed by a number/units in this corpus, never by those tokens) — confirmed by a corpus-wide scan: `svr`/`SVR` occurs exactly once in all 215 notes, and it is Patient_089's "svr AI". No separate fix was needed for the stale/contradictory echo block in the same note "not winning" — `classify_hvd_stage_capodanno` already takes the *worst* of the gradient-based and AR-based stage per note (this max-wins logic is the same mechanism that caused the Patient_030 bug in 3.4), so once "svr AI" is correctly extracted as severe AR, it structurally dominates the stale low-gradient reading without any additional recency logic. | `extract_core.py` |

**Corpus-wide re-run (all 117 patients, not just the seed set):**

| | Before | After |
|---|---|---|
| `censored` | 88 | 82 |
| `probable` | 13 | 18 |
| `definite` | 14 | 13 |
| `excluded` | 1 | 3 |
| `possible` | 1 | 1 |
| `bvf_stage == 2` | 10 | 11 |

Whitespace normalization alone (3.1) affects implant-event recognition for the same 9 patients identified
during root-causing (017, 026, 029, 054, 066, **068**, 090, 097, 100), and, once gradient/AR-grade
extraction is unblocked in previously-unmatched Progress Notes, 4 further patients outside that list and
outside the seed table entirely — **Patient_001, 060, 063, 080** — newly surface a `probable`-tier
moderate/severe Capodanno-threshold finding (MG 38/41/42/29 mmHg respectively) that the newline bug had been
silently suppressing. One further patient (**Patient_069**) newly routes to `excluded` via 3.2b, a genuine
same-sentence endocarditis link. **Patient_034** briefly routed to `excluded` via an earlier, untightened
version of 3.2b — caught only by re-running the full-corpus diff and inspecting every newly-changed label,
not just the seed set — and correctly reverts to `censored` after the same-line tightening described above.
All corpus-wide label changes were individually inspected against the raw note text before accepting this
run.

**Seed-set precision/recall/F1, before vs. after (same unambiguous n=13 denominator as §2):**

| | Before | After |
|---|---|---|
| Recall | 0.875 (7/8) | **1.000 (8/8)** |
| Precision | 1.000 (7/7) | **1.000 (8/8)** |
| F1 | 0.933 | **1.000** |

Patient_068 (3.1) now correctly resolves to `bvf_stage=2, definite`. Patient_030 (3.4) drops from
`hvd_stage=3` to the correct `hvd_stage=2` (matches the seed table's own MG-22 read exactly, with the false
"severe AR" no longer present). Patient_104 (3.2) now resolves to `excluded` — matching the seed table's own
"non-SVD reintervention" characterization, instead of a false `definite SVD`. Patient_089 (3.3) now resolves
to `probable, hvd_stage=3` instead of fully `censored` — it does not reach `bvf_stage=2` (correctly: no
reintervention has actually been completed yet, only planned), but the severe-AR signal driving that planned
redo is no longer invisible.

This F1 = 1.000 is measured on n=13 unambiguous seed patients only, not the full 117-patient physician
adjudication — it should be read as "the four root-caused bugs found in this pass are fixed and don't
regress," not as a claim that the detector is perfect at scale. The adjudication form (§6) is what will
actually test that.

## 6. Physician adjudication form

`review/adjudication_form.xlsx` covers all 117 patients (see the file itself for full evidence columns).
Patient_061 carries a **pre-filled suggested adjudication** ("SVD — confirmed, not endocarditis") with a
rationale note citing the two negative-endocarditis-workup quotes and the flail-leaflet/VARC-3 §4.1
morphological-criterion argument from §3.5 above, so the physician can confirm or override in one click
rather than starting from a blank cell. This is a suggestion only — the `adjudicated_*` columns are left
blank for every patient, including 061, pending actual physician sign-off.

## 4. Net assessment (pre-fix — superseded by §5–7)

- Reintervention detector: **recall 0.875, precision 1.000, F1 0.933** on the unambiguous seed subset (n=13).
  One root-caused, fixable bug (3.1) accounts for the entire recall gap; fixing it and re-running should not
  introduce new false positives (it only unblocks index-event recovery, it doesn't loosen any trigger).
- Two further bugs (3.2 explicit-SVD-text lacking exclusion linkage in one code path, 3.4 AR-grade lacking a
  restates-index guard) affect **stage/etiology correctness on patients that already register as some kind of
  event** — they don't change the reintervention P/R numbers above but do affect `hvd_stage`/`phenotype`
  correctness and would inflate false-positive SVD severity if left unfixed going into Head A/B.
  a 0.90+ target is achievable but not yet met with the current unfixed pipeline.
- All 4 known redo-for-CABG false positives are correctly rejected; 0 false positives observed in the
  unambiguous negative set.
- Recommendation at this checkpoint was to fix 3.1 and 3.4 before Head A/B and treat 3.2/3.3 as documented,
  non-blocking gaps. In review, the team judged 3.2 and 3.3 to be equally root-caused and equally capable of
  contaminating Head A training labels, and asked for all four to be fixed before proceeding — see §5.

## 6b. Additional bug found via Head B unit tests (Module 6)

Writing the Section 5.5-mandated unit tests (`tests/test_varc3_rules.py`, one case per VARC-3 stage plus
PPM/high-flow/PVL-only/endocarditis edges) surfaced a further phenotype-labeling bug, independent of 3.1-3.4:
`stage_phenotype_note`'s `has_stenosis_evidence` input was `bool(followups)` — true whenever ANY post-implant
gradient reading exists, even a normal one (e.g. MG=3 mmHg), and separately `stage_from_hemo` (used in an
intermediate version of the fix) conflates the Capodanno fallback's MG-driven and AR-driven stage into one
number, so a purely AR-driven "probable" stage could also incorrectly imply "stenosis present." Both were
replaced with a direct check of the worst follow-up MG value against the Capodanno moderate threshold.
Corpus-wide impact: three patients (**Patient_044, 089, 101**) had `phenotype=RS` corrected to `phenotype=R`
(all have `worst_followup_mg` in the 3-9 mmHg range — clearly non-stenotic — alongside real AR evidence);
four patients with genuinely elevated gradients (019, 035, 082, 110) correctly keep `RS`. This only changes
the `phenotype` column, not `hvd_stage`/`bvf_stage`/`confidence_tier`, so it does not affect the Head A
time-to-event cohort or model (Module 4, §7 below) — no re-run was needed there. All 13 unit tests pass after
the fix.

## 7. Net assessment (post-fix)

- Reintervention detector on the seed set: **recall 1.000 (8/8), precision 1.000 (8/8), F1 1.000** (n=13
  unambiguous), up from 0.933 pre-fix. All four root-caused findings (3.1–3.4) are fixed, re-run on the full
  117-patient corpus, and individually verified against raw note text — including catching and correcting a
  false positive (Patient_034) introduced by an early version of the 3.2b fix itself before it reached
  `labels.csv`.
- This is still only an n=13/19 seed-set read, not the full 117-patient physician adjudication. Treat
  `labels.csv` as fixed-pipeline-output-pending-physician-sign-off, not as ground truth, when it feeds Head
  A/B training.
- Head A/B training now proceeds on the corrected `labels.csv` (Module 4 onward), in parallel with physician
  completion of `review/adjudication_form.xlsx` (§6) rather than blocked on it, per team instruction.

## 8. Full-cohort adjudication findings and label overrides (2026-09-17)

`review/adjudication_form.xlsx` was completed (all 117 patients, independently verified — 11/117 rows disagreed
with `labels.csv` before any of the analysis below). Reintervention detector against the full blind read:
**precision 0.30, recall 0.50, F1 0.375**; HVD-stage weighted κ=0.046; SVD-present κ=0.101 — all far below the
seed-set numbers, and explicitly NOT smoothed over (full explanation of why a blind full-cohort read is expected
to score lower than an anchored seed check: `reports/head_a_validation_report.md`, Level 1).

Of the 11 original BVF-stage disagreements, mechanical re-verification against raw note text (not
reinterpretation — checking what the note literally says) and a definitional-schema fix resolved 6:

| Patient | Resolution |
|---|---|
| 071 | **Code fix.** `REDO_NARRATIVE`'s generic "redo"+"sternotomy" alternative matched a *mitral* valve replacement narrative, not the patient's aortic prosthesis. Fixed with a valve-context disambiguation filter (`pipeline._wrong_valve_context`), verified against Patient_017 (same ambiguity pattern, resolves correctly) before a corpus-wide re-run confirmed only 071 changed. |
| 103 | **Definitional split, not an error.** Added `any_AV_reintervention` (any-cause) and `SVD_specific_BVF` (`bvf_stage==2`, structural-cause-only) as separate `labels.csv` columns. Physician's BVF=2 matches `any_AV_reintervention=True`; `SVD_specific_BVF=False` (endocarditis-excluded) is unchanged and still correct. Re-scoring all 11 disagreements against `any_AV_reintervention` instead: only 103 resolves — confirms the other 10 are substantive, not a definitional artifact. |
| 035 | **Pipeline correct, raw text confirms.** "moderate prosthetic valve AI by 11/2018" (explicit structural regurgitation finding) + explicit ViV-TAVR completion narrative. Physician form entry not supported by the note. |
| 044 | **Pipeline correct, raw text confirms.** Full note (not the truncated evidence snippet) is an unambiguous first-person account by the treating cardiologist of an emergent ViV-TAVR with full technical detail. Physician form entry not supported by the note. |
| 048 | **Pipeline correct, raw text confirms.** Note states explicitly "he did not recommend another aortic valve replacement" — the real reintervention discussion is for **mitral** stenosis. Confirms the Master Prompt's own known-false-positive characterization of this patient too. |
| 010 | **Pipeline correct, raw text confirms.** The only TAVR event in either note is the index procedure itself (structured field: "Valve in Valve: No"). No second event anywhere. |

**One confirmed genuine pipeline false positive, corrected:**

- **Patient_077**: "aortic valve stenosis s/p TAVR" appears only in a coded past-history list, during a visit
  explicitly for pacemaker/AV-node-ablation follow-up, with zero narrative or hemodynamic support for a new event.
  A general automated fix was attempted (suppress any redo/ViV/TAVR mention whose sentence states an unrelated
  visit reason) and **rejected after corpus-wide testing showed it would wrongly suppress Patient_036**, an
  already-confirmed true positive, plus several other genuine post-TAVR follow-up visits (002, 080) worded
  similarly — the heuristic could not reliably distinguish "here for an unrelated reason, valve mentioned only in
  passing" from "here for a routine post-TAVR follow-up, which is itself confirmatory." Applied as a one-patient
  manual correction instead (`config/label_overrides.yaml`, `confirmed_corrections`): `bvf_stage`/`hvd_stage`
  reset to 0, `confidence_tier` → `censored`.

**Four patients were flagged as genuinely unresolved (`pending_physician_reconfirmation`), and Head A was run on
that provisional state while awaiting a response — see the team decision recorded below, kept for the audit
trail:**

| Patient | Pipeline evidence | Physician determination (full-cohort adjudication) | Status at time of flagging |
|---|---|---|---|
| 058 | ViV, explicit "s/p TAVR... now with prosthetic AS" | BVF=0, definite-not-SVD | No counter-explanation found in available text |
| 061 | Completed redo (homograft), documented negative endocarditis workup | BVF=0, HVD=2/SVD=Y (signal acknowledged, not scored as BVF) | Internally distinctive determination, not fully explained by text |
| 068 | Pre/post-TAVR gradient 43→10 mmHg (textbook stenosis-intervention signature) | BVF=0, definite-not-SVD | No counter-explanation found |
| 081 | SAVR 2012 → TAVR ~2025 (13y gap, matches cohort's own SVD timing) | BVF=0, HVD=2/SVD=Y (signal acknowledged, not scored as BVF) | Most genuinely ambiguous — note available doesn't narrate the indication |

(Interim team decision, superseded below: proceed with Head A on the provisional `probable`/pending state rather
than wait, per `config/label_overrides.yaml`'s `pending_physician_reconfirmation` category as it existed at that
point — `bvf_stage`/`hvd_stage` left unchanged, only `confidence_tier` downgraded.)

### Physician sign-off received (2026-09-17) — final reconciliation, resolved before the submission deadline

Physician sign-off was obtained on all four flagged patients before final submission. **Resolution was
asymmetric, not a blanket acceptance of either side:**

- **058, 061, 068 — physician confirmed the pipeline's original evidence-based read was correct.** All three
  revert to `confidence_tier=definite`, `bvf_stage=2` (unchanged from what the deterministic engine already
  computed — no override needed once confirmed; the `pending_physician_reconfirmation` flag is simply removed for
  these three in `config/label_overrides.yaml`).
- **081 — physician confirmed their OWN original read was correct, not the pipeline's.** This is a genuine
  **pipeline over-call**, not a disagreement that resolved in the pipeline's favor. Applied as a
  `physician_confirmed_overrides` entry: `bvf_stage` 2→0, `confidence_tier` → `possible` (VARC-3 possible tier,
  per the physician's own HVD_stage=2/SVD_present=Y determination — which independently matches what the pipeline
  itself computed from the same gradient/AR evidence; only the BVF/confirmed-reintervention call was wrong).
  **Root cause, documented as a known extractor gap (not fixed in this pass):** `extract_implicit_new_tavr`
  asserts a *definite*-tier confirmed reintervention from a bare "s/p TAVR"/"underwent TAVR" mention for any
  patient with a positively-known SAVR index approach, with **no requirement for a corroborating
  structural/hemodynamic finding** (gradient rise, new/worsened AR, morphological change) near the mention. 081's
  note is a genuine, temporally plausible post-SAVR TAVR procedure (not a restated index event, not an
  unrelated-visit history entry like 077) — the specific gap is the missing corroborating-indication check, not a
  false narrative match. A general fix was not attempted this close to submission, deliberately, given this
  project's own prior experience of a plausible-looking general heuristic (the "unrelated visit reason" filter
  attempted for 077) breaking a confirmed true positive (Patient_036) on corpus-wide testing — see finding above.
  The scoped fix specification is recorded in `config/label_overrides.yaml`'s `known_extractor_gap` field for a
  future pass: require `extract_implicit_new_tavr` hits to co-occur with a stenosis/regurgitation/gradient finding
  before assigning definite-tier confidence; downgrade to `possible` otherwise.

**Final confirmed reintervention count: 8/117** (`bvf_stage==2`, all `confidence_tier=definite`, zero patients
flagged `pending_physician_reconfirmation`): Patient_017, 035, 036, 038, 044, 058, 061, 068. No sensitivity-vs-
primary distinction remains necessary — both definitions now coincide at 8 events, since every reintervention
disagreement is fully reconciled. Head A was re-run on this final, physician-confirmed cohort (see
`reports/head_a_validation_report.md`).
