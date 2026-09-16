"""
Builds the PUBLIC-REPO-SAFE version of the notes NLP notebook: same code
and methodology as notes_nlp_pipeline.ipynb, but with every real clinical
note excerpt replaced by a synthetic illustrative example, per the repo's
own rule ("Do not upload real patient data or clinical notes to this
repository ... this repo may be publicly visible"). Structured, de-identified
per-patient fields (stage numbers, valve model/size, years) are kept, since
those are the actual deliverable and aren't narrative clinical text.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""\
# Component 1 — NLP Extraction & VARC-3 Phenotyping Engine (Notes)

Dyania Health Docathon, Sep 2026, Team `sarra`. Deterministic ground-truth
construction from `notes_deidentified.xlsx` (117 patients, 215 notes): a
regex/negation extraction layer feeds a rule-based VARC-3 (with Capodanno
2017 fallback) label assignment, producing the SVD/BVF labels that the
survival model (`model/approach.md`) is trained/validated against.

> **Data note:** the underlying `notes_deidentified.xlsx` / `labs_deidentified.xlsx`
> / `medications_deidentified.xlsx` files, and the raw-note physician review
> workbook, are **not included in this repository** (see the repo root
> `.gitignore` and the warning in `README.md`) — they stay within the team's
> private data-sharing channel. Everything demonstrated below either runs on
> that private data locally (not committed) or, where this notebook needed a
> worked example for documentation, uses a clearly-labelled **synthetic**
> excerpt written to illustrate the extraction pattern, not a real patient's
> text.

**Why deterministic and not learned:** with an expected 6–10 true SVD/BVF
events in this cohort, a trained classifier for the *labels themselves*
would have nothing to learn from and no way to be audited. A rule engine
built on transparent, cited VARC-3/Capodanno thresholds is fully
inspectable by the clinician on the team — every trigger is traceable to
the sentence that fired it (in the private, non-committed audit files),
which is what feeds the Level-1 blind physician review.

**Pipeline stages**
1. Load notes, build per-patient implant timelines (index date, model,
   size; pre/post-implant temporal attribution)
2. Extract clinical findings per note (gradients, EOA/DVI, AR grade, LVEF,
   redo/reoperation, valve-in-valve, endocarditis/thrombosis exclusions,
   explicit SVD/prosthetic-failure language, morphology-only findings) —
   each with negation handling
3. Aggregate per patient and apply the VARC-3 rule engine (Table 12/13,
   Capodanno 2017 fallback when no baseline echo is recoverable)
4. Sanity-check against a set of manually-reviewed reference patients, and
   export the (private, non-committed) physician blind-review workbook
""")

code("""\
import sys, pandas as pd
sys.path.insert(0, ".")  # library modules live alongside this notebook

from temporal import build_patient_timelines, years_since_index
from extract_core import (
    extract_gradients, extract_lvef, extract_ar_grade, extract_redo, extract_viv,
    extract_exclusions, extract_svd_explicit, extract_morphology,
    extract_implicit_new_tavr, sentence_window,
)
from varc3_rules import assign_patient_label
from pipeline import run_pipeline

pd.set_option("display.max_colwidth", 80)
""")

md("""\
## 1. Data grounding (methodology, not raw excerpts)

Two distinct note templates coexist in this corpus (a structured "HVI"
operative template with `Field: Value` lines, and free-text narrative op
reports), and several terms collide with unrelated boilerplate that is
identical across every patient using that EHR template — not
patient-specific content, so safe to describe here:

- `"Epic"` mostly means **the EHR system** (a fixed document header /
  "reviewed in Epic" boilerplate line) or **epicardial** pacing wires —
  only rarely the St. Jude Epic valve. Every valve-model match requires
  supporting context and blocks these collocations (`valve_dictionary.py`).
- `"Reoperation: No previous surgeries"` and `"Valve in Valve: No"` are
  **fixed template stubs**, not positive findings — a naive keyword search
  on "reoperation" or "valve in valve" would call every patient with this
  template a reintervention.
- `"Endocarditis prophylaxis therapy for dental procedures"` is routine,
  standardised preventive-care advice text, not a diagnosis.
- The dominant hemodynamics template — *"The peak gradient is X mmHg, the
  mean gradient is Y mmHg and the dimensionless valve index is Z"* — is a
  fixed report-generator sentence reused verbatim for the **mitral and
  tricuspid** valves elsewhere in the same notes, so gradient extraction
  has to resolve *which* valve a given sentence is about from context, not
  just find the number.
- Service Date is **year-only**; some structured "Past Surgical History"
  fields retain `MM/YYYY`, giving sub-year precision for a minority of
  dates. A note filed in the **same year** as the index implant is very
  often the pre-operative work-up for the native valve (why the patient
  got the prosthesis in the first place) — attributing that gradient to
  the new prosthesis was the single largest source of false positives
  during development (see section 4).

**Synthetic worked example** (illustrates the extraction, not a real
patient — everything below this point in the notebook that touches actual
note text stays in the team's private data channel, never committed here):
""")

md("""\
Two separate synthetic notes below (an index operative note and a later
follow-up note), mirroring how the real pipeline actually consumes the
data: one row per clinical encounter, each with its own Service Date —
never a single blob spanning multiple encounters.
""")

code('''\
SYNTHETIC_OP_NOTE = """
PREOPERATIVE DIAGNOSIS: Severe aortic stenosis.
OPERATIONS: Median sternotomy, aortic valve replacement with a 25-mm Trifecta GT bioprosthesis.
Reoperation: No previous surgeries
Valve in Valve: No
"""

SYNTHETIC_FOLLOWUP_NOTE = """
Follow-up note, 6 years after index AVR: patient re-presents with progressive bioprosthetic AVR stenosis. The peak gradient is 58 mmHg, the mean gradient is 34 mmHg and the dimensionless valve index is 0.21. Underwent successful valve-in-valve TAVR.
"""

print("-- index operative note --")
print("Redo/reoperation hits:", extract_redo(SYNTHETIC_OP_NOTE))
print("ViV hits:", extract_viv(SYNTHETIC_OP_NOTE))
print("(both empty as expected: 'No previous surgeries' / 'Valve in Valve: No' are negative structured template fields, not positive findings)")

print("\\n-- follow-up note, 6 years later --")
print("SVD-explicit hits:", [sentence_window(SYNTHETIC_FOLLOWUP_NOTE, h["start"], h["end"])
                              for h in extract_svd_explicit(SYNTHETIC_FOLLOWUP_NOTE)])
print("Gradients:", [(g["peak"], g["mean"], g["dvi"]) for g in extract_gradients(SYNTHETIC_FOLLOWUP_NOTE)])
print("ViV hits:", extract_viv(SYNTHETIC_FOLLOWUP_NOTE))
''')

md("## 2. Run the pipeline on the private dataset")

code("""\
# notes_deidentified.xlsx is provided to the team out-of-band (not in this
# repo — see data/data_plan.md) and is expected at this relative path when
# running the notebook locally with the private data attached.
import os
NOTES_PATH = "notes_deidentified.xlsx"
if os.path.exists(NOTES_PATH):
    labels_df, audit_df = run_pipeline(NOTES_PATH)
    print(labels_df["confidence_tier"].value_counts())
else:
    print(f"{NOTES_PATH} not found locally — this cell is a no-op in the public "
          "repo checkout. See data/data_plan.md for how to obtain the private data.")
""")

md("""\
## 3. VARC-3 / Capodanno rule engine

`varc3_rules.py` implements VARC-3 Table 12/13 (delta-gradient + concurrent
EOA/DVI drop, requires a baseline echo) as the primary path, falling back
to Capodanno 2017 absolute thresholds (MG ≥20 moderate / ≥40 severe, no
baseline required) for the majority of patients where no baseline is
recoverable — tagged **probable**, never **definite**. A redo/ViV
reintervention or explicit SVD/prosthetic-failure text is **definite**
regardless of the gradient picture. Endocarditis/thrombosis mentioned in
the same note as a reintervention trigger routes the patient to
**excluded** (competing risk) instead — see `data/data_plan.md` section 4
for the full ground-truth hierarchy and the validated result summary
(structured, de-identified counts only).
""")

code("""\
if os.path.exists(NOTES_PATH):
    print(\"BVF Stage 2 (reintervention) patients:\", (labels_df.bvf_stage == 2).sum())
    display_cols = [\"profile_key\", \"confidence_tier\", \"hvd_stage\", \"bvf_stage\",
                     \"index_implant_year\", \"index_valve_model\"]
    labels_df[labels_df.bvf_stage == 2][display_cols]
""")

md("""\
## 4. Known limitations (see `study_protocol.md` for the physician-review plan)

- **Exclusion linkage is note-level, not clause-level**: a redo/SVD trigger
  is vetoed if endocarditis/thrombosis appears anywhere in the same note.
  Correct for every case observed in development but would mis-fire if a
  future note mentioned unrelated remote endocarditis history alongside an
  unrelated SVD trigger.
- **BVF Stage 3 (valve-related death)** is not derivable from this
  notes-only extract — no structured vital-status field survives
  de-identification. The pipeline never asserts it.
- **Multiple gradient timepoints narrated within one note** are pooled to
  that note's worst reading rather than split into separate follow-up
  timepoints — under-uses information rather than over-claiming.
- **A `redo`/`reoperation` mention framed as "past medical history"** is
  suppressed (it usually restates the already-captured index implant done
  via a redo-sternotomy approach because of unrelated prior surgery) — a
  heuristic, not a certainty, and exactly the class of case the blind
  physician review is designed to catch.
- **Implicit ViV detection** ("had TAVR" with no explicit "valve-in-valve"
  wording) only fires when the patient's index implant is positively
  confirmed as SAVR, to avoid mistaking a patient's own index TAVR,
  narrated in a later note, for a second procedure.

## 5. Physician blind-review workbook (Level 1 validation)

Generated locally by `reviewer_sheet.py` — **not committed to this repo**
(it contains full raw note text). Run it against the private
`notes_deidentified.xlsx` to produce `physician_review_worksheet.xlsx` for
the team physician, following the blind-review protocol in
`study_protocol.md`.
""")

nb["cells"] = cells
nbf.write(nb, "01_notes_nlp_ground_truth.ipynb")
print("public notebook written")
