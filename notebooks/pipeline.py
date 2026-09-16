"""
End-to-end orchestration: notes_deidentified.xlsx -> per-patient VARC-3
label + reviewer worksheet. This is the "μηχανή φαινοτυπισμού με αυστηρή
επικύρωση" from the architecture (section 1): deterministic extraction +
deterministic rule engine, with every trigger traceable back to the
sentence it came from, ready for the physician's blind Level-1 review.

Known, deliberate simplifications (documented rather than hidden — flag
these to the physician reviewer and in study_protocol.md):
  1. Exclusion linkage (endocarditis/thrombosis/PVL vetoing a redo/SVD
     trigger) requires the exclusion mention to fall within
     EXCLUSION_LINK_WINDOW characters of the trigger, not merely "anywhere
     in the note" (an earlier version linked per-note, which over-triggered
     on notes where a distant, unrelated exclusion mention — e.g. remote
     endocarditis history in a comorbidity list far from the actual redo
     trigger — vetoed a genuinely SVD-driven event). This is still a
     proximity heuristic, not clause-level causal parsing, and is flagged
     in the reviewer sheet's rationale column for a human to catch either
     way.
  2. BVF Stage 3 (valve-related death) is not derivable from this
     notes-only extract (no structured vital-status field survives
     de-identification); the pipeline never asserts it.
  3. A gradient statement's "post-implant" attribution uses the *note's*
     Service Date year against the patient's index implant year. Prior/
     current gradient pairs narrated within one note (e.g. a note stating
     both a prior and a today's gradient reading in the same sentence)
     are NOT currently disambiguated to two separate follow-up
     timepoints — both are pooled into that note's "worst" reading. This
     under-uses information rather than over-claiming it.
  4. A redo/ViV mention for a patient whose index implant date could not be
     recovered (index_implant_source == 'undated') is suppressed if an
     AVR/TAVR mention sits within REDO_INDEX_ADJACENCY_WINDOW characters
     after it, on the assumption this is the patient's own (undated) index
     procedure being restated (op-report-style "Redo.\nAortic valve
     replacement..." list phrasing), not a second event. This is a
     heuristic that trades a little recall for precision specifically in
     the no-anchor case, and is exactly the ambiguity Level-1 physician
     review should adjudicate.
  5. A redo/ViV mention is also suppressed as history restatement, not a
     new event, if (a) it verbatim-matches (whitespace-normalized) a span
     inside the patient's own dated implant-event note text — an EHR
     "copy-forward" artifact where an old Operative Report's PROCEDURE
     section is pasted into a much later note — or (b) it is immediately
     followed by a de-identification placeholder standing in for a
     redacted date (e.g. "AVR in [ADDRESS]") *and* preceded in the same
     sentence by a dated CABG mention (e.g. "CABG surgery in 2008"),
     mirroring the confirmed "prior CABG in <year> and redo sternotomy /
     AVR in <year>" surgical-history template seen elsewhere in this
     corpus with both dates intact. Both are still proximity/pattern
     heuristics on de-identified, redacted text, not certainty, and are
     flagged for physician review the same way.
"""
import re

import pandas as pd

from temporal import build_patient_timelines, years_since_index, AVR_PATTERN, TAVR_PATTERN
from extract_core import (
    extract_gradients, extract_lvef, extract_ar_grade, extract_redo,
    extract_viv, extract_exclusions, extract_svd_explicit, extract_morphology,
    extract_implicit_new_tavr, extract_gradient_trend, sentence_window,
)
from sectioning import extract_date_tokens
from varc3_rules import assign_patient_label

# Exclusion mention must fall within this many characters of the
# redo/ViV/SVD trigger to veto it — roughly the same clause/sentence, not
# "anywhere in the note" (see simplification #1 above). Calibrated against
# this corpus: a genuine same-clause link ("...endocarditis s/p redo AVR...")
# sits ~5 characters apart; an unrelated remote-history mention elsewhere in
# a long problem-list note sits 250+ characters apart.
EXCLUSION_LINK_WINDOW = 200
# How close an AVR/TAVR mention must follow a redo/ViV narrative hit, for a
# patient with no dated index implant, to be treated as restating that same
# (undated) index procedure rather than a new event (simplification #4).
REDO_INDEX_ADJACENCY_WINDOW = 100


def _trigger_near_exclusion(text: str, exclusions: dict, trigger_start=None, trigger_end=None) -> str:
    """Return a short reason string if an exclusion term falls within
    EXCLUSION_LINK_WINDOW characters of the trigger span (used to veto a
    redo/SVD trigger). Falls back to "anywhere in the note" only when the
    trigger has no character position (a structured-field hit)."""
    if not exclusions:
        return None
    reasons = []
    for name, hits in exclusions.items():
        for h in hits:
            if trigger_start is not None:
                dist = max(0, max(trigger_start - h["end"], h["start"] - trigger_end, 0))
                if dist > EXCLUSION_LINK_WINDOW:
                    continue
            reasons.append(f"{name} ('{sentence_window(text, h['start'], h['end'], 30)}')")
    return "; ".join(reasons) if reasons else None


def _restates_undated_index(text: str, hit: dict, index_implant_source: str) -> bool:
    """True if this redo/ViV narrative hit is likely restating a patient's
    own (undated) index implant rather than reporting a new reintervention
    — see simplification #4. Checks the hit's own matched span (not just
    what follows it): REDO_NARRATIVE's own qualifying suffix is often
    'AVR'/'aortic valve' (e.g. "Redo... Aortic valve replacement
    utilizing..."), which already sits *inside* the match, not after it."""
    if index_implant_source != "undated" or hit.get("start") is None:
        return False
    span = text[hit["start"]:hit["end"] + REDO_INDEX_ADJACENCY_WINDOW]
    return bool(AVR_PATTERN.search(span) or TAVR_PATTERN.search(span))


def _restates_known_index_year(text: str, hit: dict, index_implant_year, window: int = 150) -> bool:
    """True if a redo/ViV narrative hit has the patient's own (already-dated)
    index implant year anywhere nearby (±window chars) — e.g. "...redo
    sternotomy with a bioprosthetic AVR ... in 2010" where 2010 is that
    patient's actual index year. Checks every date token in range (not just
    the single nearest one), since a note can carry an older, closer,
    unrelated year (e.g. a prior CABG year) alongside the index year a
    little further away. This is the patient's own index procedure being
    restated as history, not a new reintervention this note."""
    if index_implant_year is None or hit.get("start") is None:
        return False
    lo, hi = max(0, hit["start"] - window), min(len(text), hit["end"] + window)
    return any(tok["year"] == index_implant_year for tok in extract_date_tokens(text[lo:hi]))


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def _restates_index_note_text(text: str, hit: dict, index_note_texts, window: int = 120) -> bool:
    """True if the text around a redo/ViV narrative hit is a near-verbatim
    copy-forward of the patient's own dated implant-event note (observed in
    this corpus: an old Operative Report's full PROCEDURE section pasted,
    with only whitespace reflow, into a much later Progress Note) — a real,
    fairly common EHR pattern, not a new event."""
    if hit.get("start") is None or not index_note_texts:
        return False
    snippet = _normalize_ws(text[max(0, hit["start"] - 40):hit["end"] + 80])
    if len(snippet) < 30:
        return False
    return any(snippet in _normalize_ws(idx_text) for idx_text in index_note_texts)


CABG_HISTORY_PATTERN = re.compile(
    r"\b(?:coronary artery bypass graft(?:ing)?|\bCABG\b)\b[^.]{0,40}?\bin\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def _redacted_date_after_cabg_history(text: str, hit: dict, window_before: int = 150) -> bool:
    """True if a redo/ViV hit is immediately followed by a de-identification
    placeholder standing in for a redacted date (e.g. "AVR in [ADDRESS]"),
    AND is preceded — in the same history-listing sentence — by a dated
    CABG mention (e.g. "coronary artery bypass graft surgery in 2008").
    This is the same "prior CABG in <year> and redo sternotomy/AVR in
    <year>" surgical-history template that _restates_known_index_year
    already catches when both dates survive de-identification intact; here
    the second date was redacted (mistagged as an address/name rather than
    a date), so no year is available to compare directly, but the sentence
    structure and CABG-history framing make this the patient's own
    already-dated index procedure being recapped as history, not a new
    reintervention. See simplification #5."""
    if hit.get("start") is None:
        return False
    tail = text[hit["end"]:hit["end"] + 50]
    if not re.search(r"\bin\s+\[[A-Z]", tail):
        return False
    head = text[max(0, hit["start"] - window_before):hit["start"]]
    return bool(CABG_HISTORY_PATTERN.search(head))


def run_pipeline(notes_path: str):
    df = pd.read_excel(notes_path)
    df["Notes"] = df["Notes"].astype(str)
    timelines = build_patient_timelines(df)

    patient_records = []
    note_level_rows = []  # for audit / reviewer sheet evidence column

    for pid, sub in df.groupby("Profile Key"):
        tl = timelines[pid]
        index_year = tl["index_implant_year"]
        last_note_year = int(sub["Service Date"].max())
        # Full text of this patient's own implant-event note(s), for the
        # copy-forward check above.
        index_note_texts = [str(df.loc[e["row_index"], "Notes"]) for e in tl["events"]]

        ev = dict(
            has_exclusion=False, exclusion_reason=None,
            has_redo=False, redo_is_viv=False, redo_note_year=None, redo_evidence=None,
            has_svd_explicit_text=False, svd_explicit_evidence=None, svd_explicit_year=None,
            followup_gradients=[], baseline_gradient=None,
            max_ar_ordinal=None, ar_evidence=None,
            has_morphology_only=False, morphology_evidence=None,
            has_gradient_trend_only=False, gradient_trend_evidence=None,
            last_note_year=last_note_year, index_implant_year=index_year,
        )

        for idx, row in sub.iterrows():
            text = row["Notes"]
            note_year = int(row["Service Date"])
            # Strictly AFTER the index implant year, not >=: a note filed in
            # the same calendar year as the index operation is very often
            # the pre-operative work-up describing the native valve's own
            # severe stenosis (that's *why* the patient got the prosthesis),
            # not post-implant surveillance of the new one (observed in this
            # cohort: a same-year CVICU coordination-of-care note describing
            # the native valve's own pre-op severe stenosis, gradient and
            # all, filed the same calendar year as that patient's implant).
            # When the index year itself is unknown we can't apply this guard, so we
            # fall back to treating the note as potentially post-implant
            # (permissive) rather than silently discarding it.
            years_post = years_since_index(note_year, index_year)
            is_post = True if years_post is None else years_post > 0

            exclusions = extract_exclusions(text)
            redo_hits = extract_redo(text)
            viv_hits = extract_viv(text)
            # A patient with no dated index implant: a redo/ViV mention
            # immediately followed by an AVR/TAVR mention is very likely
            # that same (undated) index procedure being restated, not a new
            # event — see simplification #4.
            redo_hits = [h for h in redo_hits if not _restates_undated_index(text, h, tl["index_implant_source"])
                         and not _restates_known_index_year(text, h, index_year)
                         and not _restates_index_note_text(text, h, index_note_texts)
                         and not _redacted_date_after_cabg_history(text, h)]
            viv_hits = [h for h in viv_hits if not _restates_undated_index(text, h, tl["index_implant_source"])
                        and not _restates_known_index_year(text, h, index_year)
                        and not _restates_index_note_text(text, h, index_note_texts)
                        and not _redacted_date_after_cabg_history(text, h)]
            if is_post and tl["index_approach"] == "SAVR":
                # A bare "s/p TAVR" / "had TAVR" mention with no explicit
                # 'valve-in-valve' wording (observed in this corpus) is
                # still a reintervention when the index prosthesis was
                # surgical —
                # see extract_implicit_new_tavr docstring. Only fire when
                # the index approach is POSITIVELY known to be SAVR (not
                # merely "not TAVR"/unknown): for the many patients whose
                # own index implant WAS a TAVR that this extract never
                # captured as a formal event (index_approach is None), the
                # same "s/p TAVR" phrase is that patient simply restating
                # their own index procedure, not a new one.
                viv_hits = viv_hits + extract_implicit_new_tavr(text)
            svd_hits = extract_svd_explicit(text)
            morph_hits = extract_morphology(text)
            grad_hits = extract_gradients(text)
            grad_trend_hits = extract_gradient_trend(text)
            ar_hits = extract_ar_grade(text)

            if (redo_hits or viv_hits) and is_post:
                # A redo/ViV mention in a note strictly after the index
                # implant year — skips the index implant note itself (a
                # redo-*approach* sternotomy for the index operation, or ViV
                # boilerplate, would otherwise co-fire on that same note).
                _rv_hit = (redo_hits or viv_hits)[0]
                exclusion_reason = _trigger_near_exclusion(
                    text, exclusions, _rv_hit.get("start"), _rv_hit.get("end"))
                if exclusion_reason:
                    if not ev["has_exclusion"]:
                        ev["has_exclusion"] = True
                        ev["exclusion_reason"] = f"redo/ViV in {note_year} attributed to: {exclusion_reason}"
                else:
                    ev["has_redo"] = True
                    ev["redo_is_viv"] = ev["redo_is_viv"] or bool(viv_hits)
                    ev["redo_note_year"] = note_year if ev["redo_note_year"] is None else min(ev["redo_note_year"], note_year)
                    hit = (redo_hits or viv_hits)[0]
                    snippet = hit.get("matched_text") or hit.get("value") or ""
                    if hit.get("start") is not None:
                        snippet = sentence_window(text, hit["start"], hit["end"])
                    ev["redo_evidence"] = f"[{row['Type']} {note_year}] {snippet}"

            if svd_hits and is_post:
                _svd_hit = svd_hits[0]
                svd_exclusion_reason = _trigger_near_exclusion(
                    text, exclusions, _svd_hit.get("start"), _svd_hit.get("end"))
                if svd_exclusion_reason:
                    if not ev["has_exclusion"]:
                        ev["has_exclusion"] = True
                        ev["exclusion_reason"] = f"explicit SVD-language note ({note_year}) also names: {svd_exclusion_reason}"
                else:
                    ev["has_svd_explicit_text"] = True
                    ev["svd_explicit_year"] = note_year if ev["svd_explicit_year"] is None else min(ev["svd_explicit_year"], note_year)
                    snip = sentence_window(text, svd_hits[0]["start"], svd_hits[0]["end"])
                    ev["svd_explicit_evidence"] = f"[{row['Type']} {note_year}] {snip}"

            if grad_hits:
                worst = max(grad_hits, key=lambda g: g["mean"] or 0)
                record = dict(mean=worst["mean"], peak=worst["peak"], dvi=worst["dvi"], note_year=note_year)
                if is_post:
                    ev["followup_gradients"].append(record)
                elif ev["baseline_gradient"] is None or note_year < (ev["baseline_gradient"].get("note_year") or note_year):
                    ev["baseline_gradient"] = dict(mean=worst["mean"], dvi=worst["dvi"], note_year=note_year)

            if ar_hits and is_post:
                worst_ar = max(ar_hits, key=lambda a: a["ordinal"])
                if ev["max_ar_ordinal"] is None or worst_ar["ordinal"] > ev["max_ar_ordinal"]:
                    ev["max_ar_ordinal"] = worst_ar["ordinal"]
                    ev["ar_evidence"] = f"[{row['Type']} {note_year}] {sentence_window(text, worst_ar['start'], worst_ar['end'])}"

            if morph_hits and is_post and not grad_hits:
                ev["has_morphology_only"] = True
                ev["morphology_evidence"] = f"[{row['Type']} {note_year}] {sentence_window(text, morph_hits[0]['start'], morph_hits[0]['end'])}"

            if grad_trend_hits and is_post and not grad_hits:
                # Qualitative-only worsening ("increased AVR gradients", no
                # number given) — no VARC-3 stage can be assigned, but this
                # is still a 'possible' signal for physician review rather
                # than a silent censor.
                ev["has_gradient_trend_only"] = True
                ev["gradient_trend_evidence"] = f"[{row['Type']} {note_year}] {sentence_window(text, grad_trend_hits[0]['start'], grad_trend_hits[0]['end'])}"

            note_level_rows.append(dict(
                profile_key=pid, note_type=row["Type"], service_date=note_year,
                is_post_implant=is_post, n_redo=len(redo_hits), n_viv=len(viv_hits),
                n_svd_explicit=len(svd_hits), n_gradients=len(grad_hits),
                n_exclusion=sum(len(v) for v in exclusions.values()), n_morphology=len(morph_hits),
            ))

        label = assign_patient_label(ev)
        patient_records.append(dict(
            profile_key=pid,
            index_implant_year=index_year,
            index_implant_source=tl["index_implant_source"],
            index_valve_model=tl["index_valve_model"],
            index_valve_size_mm=tl["index_valve_size_mm"],
            n_implant_events=len(tl["events"]),
            last_note_year=last_note_year,
            years_followup=(last_note_year - index_year) if index_year is not None else None,
            hvd_stage=label["hvd_stage"],
            bvf_stage=label["bvf_stage"],
            phenotype=label["phenotype"],
            confidence_tier=label["confidence_tier"],
            rationale=" | ".join(label["rationale"]),
            redo_evidence=ev["redo_evidence"],
            svd_explicit_evidence=ev["svd_explicit_evidence"],
            exclusion_reason=ev["exclusion_reason"],
            ar_evidence=ev["ar_evidence"],
            morphology_evidence=ev["morphology_evidence"],
            gradient_trend_evidence=ev["gradient_trend_evidence"],
            n_followup_gradients=len(ev["followup_gradients"]),
            worst_followup_mg=(max((g["mean"] for g in ev["followup_gradients"]), default=None)),
            baseline_mg=(ev["baseline_gradient"] or {}).get("mean"),
        ))

    labels_df = pd.DataFrame(patient_records).sort_values("profile_key").reset_index(drop=True)
    notes_audit_df = pd.DataFrame(note_level_rows)
    return labels_df, notes_audit_df


if __name__ == "__main__":
    labels_df, audit_df = run_pipeline("notes_deidentified.xlsx")
    labels_df.to_csv("labels.csv", index=False)
    audit_df.to_csv("notes_audit.csv", index=False)
    print(labels_df["confidence_tier"].value_counts())
    print()
    print(labels_df[labels_df["confidence_tier"].isin(["definite", "probable"])]
          [["profile_key", "confidence_tier", "hvd_stage", "bvf_stage", "rationale"]]
          .to_string())
