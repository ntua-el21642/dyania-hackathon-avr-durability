import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""\
# Component 1 — NLP Extraction & VARC-3 Phenotyping Engine (Notes)

Dyania Health Docathon, Sep 2026. This notebook builds and validates the
**deterministic phenotyping engine** described in the team's modelling
architecture: regex/negation extraction over `notes_deidentified.xlsx`
(117 patients, 215 notes) feeding a rule-based VARC-3 (with Capodanno 2017
fallback) label assignment, producing the ground truth that Head A
(Bayesian Weibull AFT survival model) will be trained/validated against.

**Why deterministic and not learned:** with an expected 6–10 true SVD/BVF
events in this cohort, a trained classifier for the *labels themselves*
would have nothing to learn from and no way to be audited. A rule engine
built on transparent, cited VARC-3/Capodanno thresholds is fully
inspectable by the clinician on the team, and every trigger below carries
the exact sentence it fired on — this is what feeds the Level-1 blind
physician review (`physician_review_worksheet.xlsx`), which is the real
validation of label quality (architecture doc section 6).

**Pipeline stages**
1. Load notes, build per-patient implant timelines (index date, model,
   size; pre/post-implant temporal attribution)
2. Extract clinical findings per note (gradients, EOA/DVI, AR grade, LVEF,
   redo/reoperation, valve-in-valve, endocarditis/thrombosis exclusions,
   explicit SVD/prosthetic-failure language, morphology-only findings) —
   each with negation handling
3. Aggregate per patient and apply the VARC-3 rule engine (Table 12/13,
   Capodanno 2017 fallback when no baseline echo is recoverable)
4. Sanity-check against the 7 patients manually reviewed during architecture
   design, and export the physician blind-review workbook
""")

code("""\
import sys, pandas as pd
sys.path.insert(0, ".")  # nlp_pipeline modules live alongside this notebook

from temporal import build_patient_timelines, years_since_index
from extract_core import (
    extract_gradients, extract_lvef, extract_ar_grade, extract_redo, extract_viv,
    extract_exclusions, extract_svd_explicit, extract_morphology,
    extract_implicit_new_tavr, sentence_window,
)
from varc3_rules import assign_patient_label
from pipeline import run_pipeline
from reviewer_sheet import build_reviewer_workbook

pd.set_option("display.max_colwidth", 120)
""")

md("""\
## 1. Data grounding

Before writing a single regex, we pulled real examples from the corpus to
see what the notes actually look like — this matters because two
distinct note templates coexist here (a structured "HVI" operative
template with `Field: Value` lines, and free-text "EPIC CARE" narrative
op reports), and several terms collide with unrelated boilerplate:

- `"Epic"` mostly means **the EHR system** ("Labs and medications reviewed
  in Epic", "EPIC CARE OPERATIVE REPORT" header) or **epicardial** pacing
  wires — only rarely the St. Jude Epic valve. Every valve-model match
  requires supporting context and blocks these collocations
  (`valve_dictionary.py`).
- `"Reoperation: No previous surgeries"` and `"Valve in Valve: No"` are
  **template stubs**, not positive findings — a naive keyword search on
  "reoperation" or "valve in valve" would call every patient with this
  template a reintervention.
- `"Endocarditis prophylaxis therapy for dental procedures"` is routine
  preventive-care advice, not a diagnosis.
- The dominant hemodynamics template — *"The peak gradient is X mmHg, the
  mean gradient is Y mmHg and the dimensionless valve index is Z"* — is
  reused verbatim for the **mitral and tricuspid** valves elsewhere in the
  same notes (Patient_042's MitraClip follow-up), so gradient extraction
  has to resolve *which* valve a given sentence is about from context, not
  just find the number.
- Service Date is **year-only**; some structured "Past Surgical History"
  fields retain `MM/YYYY`, which gives sub-year precision for a minority
  of dates. A note filed in the **same year** as the index implant is very
  often the pre-operative work-up for the native valve (why the patient
  got the prosthesis in the first place) — attributing that gradient to
  the new prosthesis was the single largest source of false positives
  during development (see section 4).
""")

code("""\
notes_df = pd.read_excel("notes_deidentified.xlsx")
notes_df["Notes"] = notes_df["Notes"].astype(str)
print(notes_df.shape, notes_df["Profile Key"].nunique(), "patients")
notes_df["Type"].value_counts()
""")

md("## 2. Implant timelines — index date, model, size, pre/post attribution")

code("""\
timelines = build_patient_timelines(notes_df)

reference_patients = ["Patient_071", "Patient_038", "Patient_035", "Patient_036",
                       "Patient_044", "Patient_042", "Patient_103"]
for pid in reference_patients:
    t = timelines[pid]
    print(f"{pid}: index_year={t['index_implant_year']} "
          f"(source={t['index_implant_source']}, approach={t['index_approach']}), "
          f"model={t['index_valve_model']}, size={t['index_valve_size_mm']}mm, "
          f"n_events={len(t['events'])}")
""")

md("""\
## 3. Per-note extraction — worked examples

Three examples that motivated specific guards in `extract_core.py`:
1. **Patient_071** — explicit SVD text + a genuine redo (double AVR+MVR
   root replacement)
2. **Patient_042** — the mitral/tricuspid gradient-template collision, and
   the "endocarditis prophylaxis" false positive
3. **Patient_103** — a redo driven by **endocarditis**, which must be
   *excluded* from the SVD/BVF label, not counted as one
""")

code("""\
def show_extractions(pid):
    print("=" * 100)
    print(pid)
    sub = notes_df[notes_df["Profile Key"] == pid]
    for _, row in sub.iterrows():
        t = row["Notes"]
        redo, viv, excl = extract_redo(t), extract_viv(t), extract_exclusions(t)
        svd, grads = extract_svd_explicit(t), extract_gradients(t)
        if not (redo or viv or excl or svd or grads):
            continue
        print(f"-- {row['Type']} {row['Service Date']} --")
        for h in redo:
            snip = sentence_window(t, h["start"], h["end"]) if h.get("start") is not None else h.get("value")
            print("  REDO:", snip)
        for h in viv:
            snip = sentence_window(t, h["start"], h["end"]) if h.get("start") is not None else h.get("value")
            print("  ViV :", snip)
        for name, hits in excl.items():
            for h in hits:
                print(f"  EXCLUSION ({name}):", sentence_window(t, h["start"], h["end"]))
        for h in svd:
            print("  SVD-explicit:", sentence_window(t, h["start"], h["end"]))
        for g in grads:
            print(f"  gradient: peak={g['peak']} mean={g['mean']} DVI={g['dvi']}")

for pid in ["Patient_071", "Patient_042", "Patient_103"]:
    show_extractions(pid)
""")

md("""\
## 4. VARC-3 / Capodanno rule engine

`varc3_rules.py` implements VARC-3 Table 12/13 (delta-gradient + concurrent
EOA/DVI drop, requires a baseline echo) as the primary path, falling back
to Capodanno 2017 absolute thresholds (MG ≥20 moderate / ≥40 severe, no
baseline required) for the majority of patients where no baseline is
recoverable — tagged **probable**, never **definite**, exactly as
specified in the architecture. A redo/ViV reintervention or explicit
SVD/prosthetic-failure text is **definite** regardless of the gradient
picture. Endocarditis/thrombosis mentioned in the same note as a
reintervention trigger routes the patient to **excluded** (competing risk)
instead.
""")

code("""\
labels_df, audit_df = run_pipeline("notes_deidentified.xlsx")
labels_df.to_csv("labels.csv", index=False)
audit_df.to_csv("notes_audit.csv", index=False)
labels_df["confidence_tier"].value_counts()
""")

code("""\
print("BVF Stage 2 (reintervention) patients:", (labels_df.bvf_stage == 2).sum())
labels_df[labels_df.bvf_stage == 2][["profile_key", "confidence_tier", "hvd_stage", "index_implant_year"]]
""")

md("""\
## 5. Sanity check against the manual review

These 7 patients were reviewed by hand while designing the architecture
(see conversation log / `study_protocol.md`). The pipeline is expected to
reproduce every one of them; any mismatch here is a pipeline bug, not a
matter for physician adjudication.
""")

code("""\
expected = {
    "Patient_071": "SVD / BVF Stage 2 (redo AVR+MVR, root replacement)",
    "Patient_038": "BVF Stage 2 (ViV TAVR after progressive bioprosthetic stenosis)",
    "Patient_035": "BVF Stage 2 (developed prosthetic valve stenosis, then TAVR)",
    "Patient_036": "BVF Stage 2 (ViV TAVR)",
    "Patient_044": "BVF Stage 2 (ViV TAVR, emergent, prosthetic AV failure)",
    "Patient_042": "Probable/definite HVD (explicit 'moderate prosthetic aortic valve stenosis')",
    "Patient_103": "EXCLUDED — redo AVR driven by prosthetic valve endocarditis, not SVD",
}
cols = ["profile_key", "confidence_tier", "hvd_stage", "bvf_stage", "exclusion_reason"]
check = labels_df[labels_df.profile_key.isin(expected)][cols].set_index("profile_key")
check["expected"] = pd.Series(expected)
check
""")

md("""\
## 6. Known limitations (for `study_protocol.md` / physician review)

- **Exclusion linkage is note-level, not clause-level**: a redo/SVD
  trigger is vetoed if endocarditis/thrombosis appears *anywhere in the
  same note*. Correct for every case observed here (Patient_103's redo
  note names endocarditis directly) but would mis-fire if a future note
  mentioned unrelated remote endocarditis history alongside an unrelated
  SVD trigger.
- **BVF Stage 3 (valve-related death)** is not derivable from this
  notes-only extract — no structured vital-status field survives
  de-identification. The pipeline never asserts it.
- **Multiple gradient timepoints narrated within one note** ("Prior
  peak/mean gradients were 17/10 ... today's peak gradient is 25") are
  pooled to that note's worst reading rather than split into separate
  follow-up timepoints — under-uses information rather than over-claiming.
- **A `redo`/`reoperation` mention framed as "past medical history"** is
  suppressed (it usually restates the already-captured index implant done
  via a redo-sternotomy approach because of unrelated prior surgery) —
  this is a heuristic, not a certainty, and is exactly the class of case
  the blind physician review should catch if it's wrong in either
  direction.
- **Implicit ViV detection** ("had TAVR" with no explicit "valve-in-valve"
  wording) only fires when the patient's index implant is *positively*
  confirmed as SAVR from an Operative Report — deliberately not when the
  index approach is merely unknown, to avoid mistaking a patient's own
  index TAVR, narrated in a later note, for a second procedure.

## 7. Physician blind-review workbook (Level 1 validation)

Generates `physician_review_worksheet.xlsx`: raw notes per patient +
a blank determination form (SVD Y/N, HVD stage, phenotype, BVF stage,
confidence, exclusion) to be filled in *before* looking at this notebook's
`NLP_predictions` sheet — see architecture doc section 6.
""")

code("""\
path = build_reviewer_workbook(notes_df, labels_df, "physician_review_worksheet.xlsx")
print("wrote", path)
""")

nb["cells"] = cells
nbf.write(nb, "notes_nlp_pipeline.ipynb")
print("notebook written")
