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
    valve_context_at,
)
from sectioning import extract_date_tokens, repair_line_wraps
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


def _trigger_near_exclusion_same_line(text: str, exclusions: dict, trigger_start: int, trigger_end: int) -> str:
    """Same as _trigger_near_exclusion, but additionally requires no newline
    between the trigger and the exclusion mention — i.e. they must sit in
    the same narrative sentence/line, not merely be nearby entries in a
    bulleted/tabular coded problem list. Used only for the bare-AVR/TAVR
    exclusion check below (see finding 3.2 in review/error_catalogue.md):
    an early version fired on Patient_034, whose "Aortic Valve Replacement"
    and an unrelated "Acute Deep Vein Thrombosis" are two separate,
    unconnected lines in the same coded problem list (146 chars apart, no
    causal link) — unlike Patient_104's single HPI sentence "presenting for
    TAVR +/- paravalvular leak repair", which has no newline between the
    two terms. The redo/ViV/SVD-explicit exclusion checks don't need this
    extra guard because their own narrative-keyword requirement already
    anchors the proximity heuristic to a real clinical statement; a bare
    AVR/TAVR mention has no such anchor and needs the stricter same-line
    requirement instead."""
    if not exclusions:
        return None
    reasons = []
    for name, hits in exclusions.items():
        for h in hits:
            dist = max(0, max(trigger_start - h["end"], h["start"] - trigger_end, 0))
            if dist > EXCLUSION_LINK_WINDOW:
                continue
            lo, hi = sorted((trigger_start, h["start"]))
            if "\n" in text[lo:hi]:
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


AORTIC_SELF_EVIDENT = re.compile(r"aortic|\bAVR\b|\bAV\b|\bTAVR\b", re.IGNORECASE)


def _wrong_valve_context(text: str, hit: dict) -> bool:
    """True if a redo/ViV narrative hit's own matched text carries no
    aortic-specific keyword AND the nearest valve-identity context (before
    or after, whichever is closer — see extract_core.valve_context_at)
    is a DIFFERENT valve (mitral/tricuspid/pulmonic). REDO_NARRATIVE's
    generic alternatives ("redo"+"sternotomy"/"re-operation") match on
    the surgical *approach*, not on which valve was actually operated on
    — confirmed on Patient_071: "Redo median sternotomy, mitral valve
    replacement with 27-mm Biocor bioprosthesis..." matches REDO_NARRATIVE
    via the generic "sternotomy" alternative while explicitly describing a
    MITRAL valve procedure. Checked empirically against Patient_017 (whose
    own genuinely-aortic redo/BioBentall assessment appears twice in one
    note, once with a mitral-regurgitation mention closer than any aortic
    one, once with an aortic-regurgitation mention closer) before adding
    this filter, specifically to confirm a per-hit filter — not a
    per-note one — correctly keeps 017's aortic-context occurrence while
    only dropping the ambiguous one. See review/error_catalogue.md."""
    if hit.get("start") is None:
        return False
    matched = hit.get("matched_text") or ""
    if AORTIC_SELF_EVIDENT.search(matched):
        return False
    return valve_context_at(text, hit["start"], hit["end"]) == "other"


def run_pipeline(notes_path: str):
    df = pd.read_excel(notes_path)
    # Repair mid-phrase line wraps from the source text conversion BEFORE
    # any extraction — every extractor below (implant-event detection,
    # redo/ViV, gradients, AR grade, SVD-explicit-text) matches literal-
    # space phrases and silently misses a match split across a newline
    # (see review/error_catalogue.md, finding 3.1). Applied once here so
    # every downstream consumer (build_patient_timelines and this
    # function's own note loop) sees the same repaired text and character
    # offsets stay self-consistent.
    df["Notes"] = df["Notes"].astype(str).apply(repair_line_wraps)
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
            has_any_reintervention=False, any_reintervention_evidence=None, any_reintervention_note_year=None,
            has_svd_explicit_text=False, svd_explicit_evidence=None, svd_explicit_year=None,
            has_svd_history_list_only=False, svd_history_list_evidence=None,
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
                         and not _redacted_date_after_cabg_history(text, h)
                         and not _wrong_valve_context(text, h)]
            viv_hits = [h for h in viv_hits if not _restates_undated_index(text, h, tl["index_implant_source"])
                        and not _restates_known_index_year(text, h, index_year)
                        and not _restates_index_note_text(text, h, index_note_texts)
                        and not _redacted_date_after_cabg_history(text, h)
                        and not _wrong_valve_context(text, h)]
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
            # An AR-grade mention restating the patient's own index
            # indication (e.g. "AVR for severe AI (09/2011)" naming that
            # patient's own index year) is history, not a new post-implant
            # finding — same restates-index guard already applied to
            # redo/ViV hits above, now also applied to AR grade (see
            # review/error_catalogue.md, finding 3.4: this exact pattern
            # inflated Patient_030's HVD stage from 2 to 3).
            ar_hits = [h for h in ar_hits if not _restates_known_index_year(text, h, index_year)
                       and not _restates_index_note_text(text, h, index_note_texts)]

            if (redo_hits or viv_hits) and is_post:
                # A redo/ViV mention in a note strictly after the index
                # implant year — skips the index implant note itself (a
                # redo-*approach* sternotomy for the index operation, or ViV
                # boilerplate, would otherwise co-fire on that same note).
                _rv_hit = (redo_hits or viv_hits)[0]
                # A genuine redo/ViV narrative hit means an actual aortic
                # valve reintervention procedure is described in the notes,
                # REGARDLESS of whether it turns out to be SVD- or
                # non-SVD-attributed below (Master Prompt discussion,
                # 2026-09-17: physician full-cohort adjudication raised the
                # possibility that some BVF_stage disagreements reflect
                # "any reintervention occurred" (this field) vs. "an
                # SVD-specific reintervention occurred" (has_redo /
                # bvf_stage==2 below) being two different questions, not
                # pipeline errors — Patient_103's redo-AVR-for-endocarditis
                # is the clearest example: an aortic valve WAS replaced
                # again, but not for a structural reason).
                if not ev["has_any_reintervention"]:
                    ev["has_any_reintervention"] = True
                    ev["any_reintervention_note_year"] = note_year
                    _any_snip = _rv_hit.get("matched_text") or _rv_hit.get("value") or ""
                    if _rv_hit.get("start") is not None:
                        _any_snip = sentence_window(text, _rv_hit["start"], _rv_hit["end"])
                    ev["any_reintervention_evidence"] = f"[{row['Type']} {note_year}] {_any_snip}"
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

            # A bare coded problem-list phrase ("Prosthetic aortic valve
            # failure" on its own Diagnosis/Date-table line, with no
            # narrative elaboration) cannot itself distinguish SVD from
            # PVL/endocarditis/thrombosis etiology — prefer a narrative hit
            # if this note has one, and only fall back to a bare-history
            # hit (routed to the 'possible' tier, never 'definite') when it
            # doesn't (see review/error_catalogue.md, finding 3.2).
            svd_narrative_hits = [h for h in svd_hits if not h.get("in_bare_history_list")]
            svd_bare_hits = [h for h in svd_hits if h.get("in_bare_history_list")]

            if svd_narrative_hits and is_post:
                _svd_hit = svd_narrative_hits[0]
                svd_exclusion_reason = _trigger_near_exclusion(
                    text, exclusions, _svd_hit.get("start"), _svd_hit.get("end"))
                if svd_exclusion_reason:
                    if not ev["has_exclusion"]:
                        ev["has_exclusion"] = True
                        ev["exclusion_reason"] = f"explicit SVD-language note ({note_year}) also names: {svd_exclusion_reason}"
                else:
                    ev["has_svd_explicit_text"] = True
                    ev["svd_explicit_year"] = note_year if ev["svd_explicit_year"] is None else min(ev["svd_explicit_year"], note_year)
                    snip = sentence_window(text, _svd_hit["start"], _svd_hit["end"])
                    ev["svd_explicit_evidence"] = f"[{row['Type']} {note_year}] {snip}"
            elif svd_bare_hits and is_post and not ev["has_svd_history_list_only"]:
                _bare_hit = svd_bare_hits[0]
                ev["has_svd_history_list_only"] = True
                snip = sentence_window(text, _bare_hit["start"], _bare_hit["end"])
                ev["svd_history_list_evidence"] = f"[{row['Type']} {note_year}] {snip}"

            # A bare AVR/TAVR mention (no redo/ViV keyword) with no redo/ViV
            # hit of its own can still BE the patient's actual
            # reintervention when the note explicitly frames it as driven
            # by a non-structural cause nearby — e.g. "presenting for TAVR
            # +/- paravalvular leak repair" (Patient_104: the real
            # procedure the coded-history phrase above was masking). We
            # don't create a BVF-Stage-2 SVD event from a bare AVR/TAVR
            # mention alone (too weak on its own), but the exclusion must
            # still register so this patient is routed to 'excluded'
            # rather than falling through to a weaker positive SVD signal
            # (see review/error_catalogue.md, finding 3.2).
            if is_post and not (redo_hits or viv_hits) and exclusions and not ev["has_exclusion"]:
                _avr_or_tavr = AVR_PATTERN.search(text) or TAVR_PATTERN.search(text)
                if _avr_or_tavr:
                    bare_exclusion_reason = _trigger_near_exclusion_same_line(
                        text, exclusions, _avr_or_tavr.start(), _avr_or_tavr.end())
                    if bare_exclusion_reason:
                        ev["has_exclusion"] = True
                        ev["exclusion_reason"] = f"AVR/TAVR mention in {note_year} attributed to: {bare_exclusion_reason}"
                        if not ev["has_any_reintervention"]:
                            ev["has_any_reintervention"] = True
                            ev["any_reintervention_note_year"] = note_year
                            ev["any_reintervention_evidence"] = (
                                f"[{row['Type']} {note_year}] "
                                f"{sentence_window(text, _avr_or_tavr.start(), _avr_or_tavr.end())}")

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
            index_approach=tl["index_approach"],
            n_implant_events=len(tl["events"]),
            last_note_year=last_note_year,
            years_followup=(last_note_year - index_year) if index_year is not None else None,
            hvd_stage=label["hvd_stage"],
            bvf_stage=label["bvf_stage"],
            # Two deliberately separate columns, not one (2026-09-17 team
            # discussion, prompted by full-cohort physician adjudication
            # disagreements that turned out to be a definitional question,
            # not a pipeline error -- see review/error_catalogue.md):
            #   any_AV_reintervention: a redo/ViV/AVR-TAVR-reintervention
            #     narrative was found post-implant, REGARDLESS of cause
            #     (includes endocarditis/PVL/thrombosis-attributed events).
            #   SVD_specific_BVF: bvf_stage==2, i.e. an aortic valve
            #     reintervention attributed specifically to a STRUCTURAL
            #     (non-excluded) cause -- the Master Prompt's own primary
            #     endpoint definition. Every excluded/competing-event
            #     patient has any_AV_reintervention=True (a reintervention
            #     happened) but SVD_specific_BVF=False (not for SVD)
            #     whenever the underlying trigger was itself a genuine
            #     redo/ViV/AVR-TAVR narrative, not just a bare exclusion
            #     mention with no procedure evidence at all.
            any_AV_reintervention=bool(ev["has_any_reintervention"]),
            SVD_specific_BVF=bool(label["bvf_stage"] == 2),
            any_reintervention_note_year=ev["any_reintervention_note_year"],
            any_reintervention_evidence=ev["any_reintervention_evidence"],
            phenotype=label["phenotype"],
            confidence_tier=label["confidence_tier"],
            rationale=" | ".join(label["rationale"]),
            redo_note_year=ev["redo_note_year"],
            svd_explicit_year=ev["svd_explicit_year"],
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
    labels_df = apply_label_overrides(labels_df)
    notes_audit_df = pd.DataFrame(note_level_rows)
    return labels_df, notes_audit_df


def apply_label_overrides(labels_df: pd.DataFrame, overrides_path: str = "config/label_overrides.yaml") -> pd.DataFrame:
    """Apply the manual overrides documented in config/label_overrides.yaml
    — see that file's header for the full rationale and the distinction
    between categories:
      confirmed_corrections: our own raw-text re-check conclusively
        contradicted the pipeline's automated label.
      physician_confirmed_overrides: physician sign-off confirmed the
        pipeline over-called the event; the physician's own determination
        is applied as final.
    Adds `pending_physician_reconfirmation` (bool, all False as of the
    2026-09-17 sign-off — kept as a column rather than removed so a prior
    labels.csv snapshot and the current one stay schema-compatible) and
    `physician_confirmed_override_note` (why, for the one patient where
    the physician's value replaced the pipeline's)."""
    import yaml

    labels_df = labels_df.copy()
    labels_df["pending_physician_reconfirmation"] = False
    labels_df["pending_reconfirmation_note"] = None
    labels_df["physician_confirmed_override_note"] = None

    with open(overrides_path, encoding="utf-8") as f:
        overrides = yaml.safe_load(f)

    for pid, spec in (overrides.get("confirmed_corrections") or {}).items():
        mask = labels_df["profile_key"] == pid
        if not mask.any():
            continue
        labels_df.loc[mask, "bvf_stage"] = spec["new_bvf_stage"]
        labels_df.loc[mask, "hvd_stage"] = spec["new_hvd_stage"]
        labels_df.loc[mask, "phenotype"] = spec["new_phenotype"]
        labels_df.loc[mask, "confidence_tier"] = spec["new_confidence_tier"]
        labels_df.loc[mask, "rationale"] = labels_df.loc[mask, "rationale"] + " | " + spec["rationale_note"].strip()

    for pid, spec in (overrides.get("physician_confirmed_overrides") or {}).items():
        mask = labels_df["profile_key"] == pid
        if not mask.any():
            continue
        labels_df.loc[mask, "bvf_stage"] = spec["new_bvf_stage"]
        labels_df.loc[mask, "confidence_tier"] = spec["new_confidence_tier"]
        labels_df.loc[mask, "physician_confirmed_override_note"] = spec["reason"].strip()
        labels_df.loc[mask, "rationale"] = labels_df.loc[mask, "rationale"] + " | " + spec["rationale_note"].strip()

    return labels_df


if __name__ == "__main__":
    labels_df, audit_df = run_pipeline("notes_deidentified.xlsx")
    labels_df.to_csv("labels.csv", index=False)
    audit_df.to_csv("notes_audit.csv", index=False)
    print(labels_df["confidence_tier"].value_counts())
    print()
    print(labels_df[labels_df["confidence_tier"].isin(["definite", "probable"])]
          [["profile_key", "confidence_tier", "hvd_stage", "bvf_stage", "rationale"]]
          .to_string())
