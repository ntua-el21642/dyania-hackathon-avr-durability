"""
Implant-date anchoring and pre/post-implant temporal attribution.

Architecture doc section 1: "Χρειαζόμαστε χρονική απόδοση: για κάθε τιμή
πρέπει να ξέρουμε αν είναι πριν ή μετά την εμφύτευση." Section 3: dates are
year-precision only (Service Date on each note), so the event time is
interval-censored to +/-1 year around the note year; a patient whose only
implant-date evidence is a redacted "[DATE]" token is flagged
`implant_year_source = 'undated'` and is usable for label validation but not
for time-to-event estimation (per the user's stated rule).

This module identifies, per patient:
  - every candidate "implant event" (an Operative Report / Procedures note
    that describes an aortic valve replacement, surgical or transcatheter)
  - which one is the INDEX implant (the first prosthesis) vs. a ViV
    (valve-in-valve) re-intervention
  - the anchoring year for pre/post attribution of every other note
"""
import re
from valve_dictionary import find_valve_mentions, nearest_size_mm
from sectioning import extract_date_tokens, parse_field_lines, nearest_date_year

AVR_PATTERN = re.compile(
    r"aortic valve replacement|\bAVR\b|replacement of the aortic valve|"
    r"replacement.{0,20}aortic valve",
    re.IGNORECASE,
)
TAVR_PATTERN = re.compile(
    r"\bTAVR\b|transcatheter aortic valve replacement|\bTF[- ]TAVR\b|"
    r"\bTA[- ]TAVR\b",
    re.IGNORECASE,
)
VIV_PATTERN = re.compile(
    r"valve[- ]in[- ]valve|\bViV\b|\bVIV[- ]?TAVR\b", re.IGNORECASE
)
SAVR_APPROACH_PATTERN = re.compile(
    r"median sternotomy|ministernotomy|mini-sternotomy|redo median sternotomy",
    re.IGNORECASE,
)
IMPLANT_TYPES = {"Operative Report", "Procedures"}


def classify_note_as_implant_event(row_type: str, text: str):
    """Return None if this note doesn't describe an aortic valve implant
    event, else a dict describing the event found in it."""
    if row_type not in IMPLANT_TYPES:
        return None
    has_avr = bool(AVR_PATTERN.search(text))
    has_tavr = bool(TAVR_PATTERN.search(text))
    if not (has_avr or has_tavr):
        return None
    is_viv = bool(VIV_PATTERN.search(text))
    if has_tavr:
        approach = "ViV-TAVR" if is_viv else "TAVR"
    else:
        approach = "ViV-SAVR" if is_viv else "SAVR"
    mentions = find_valve_mentions(text)
    valve_model, valve_size = None, None
    if mentions:
        first = mentions[0]
        valve_model = first["name"]
        valve_size = nearest_size_mm(text, first["start"])
    # Structured template field wins over the narrative window search when
    # present (observed in this corpus: an HVI op report template with
    # "Tissue Implant Type: X" / "Implant Size: Y" as separate template
    # lines, not adjacent in the narrative text).
    fields = parse_field_lines(text)
    if "implant size" in fields and fields["implant size"].strip().isdigit():
        valve_size = int(fields["implant size"].strip())
    if "tissue implant type" in fields and fields["tissue implant type"].strip():
        valve_model = fields["tissue implant type"].strip()
    return dict(approach=approach, is_viv=is_viv, valve_model=valve_model,
                valve_size_mm=valve_size, source="op_report")


NARRATIVE_IMPLANT_DATE_WINDOW = 60


def narrative_implant_mentions(text: str):
    """Fallback for patients with no qualifying Operative Report/Procedures
    note in this extract: scan any note's narrative for an implant mention
    ('s/p TAVR 2019', 'bioprosthetic AVR [DATE]') paired with a nearby
    explicit date token. Lower-confidence than an op report's own Service
    Date, so callers must tag these `source='narrative_mention'`."""
    out = []
    for pat, tavr in ((AVR_PATTERN, False), (TAVR_PATTERN, True)):
        for m in pat.finditer(text):
            date = nearest_date_year(text, m.start(), window=NARRATIVE_IMPLANT_DATE_WINDOW)
            if date is None or date["year"] is None:
                continue
            # Guard against grabbing a date from an unrelated adjacent line in
            # a tabular Past-[Surgical-]History list (observed in this corpus:
            # an unrelated earlier procedure's date sitting on the line right
            # above the "AORTIC VALVE REPLACEMENT [DATE]" entry): reject if a
            # newline separates the mention from the date token, unless the
            # date token is itself inline with the mention (e.g. "TAVR 2019,").
            lo, hi = sorted((m.start(), date["start"]))
            if "\n" in text[lo:hi]:
                continue
            is_viv = bool(VIV_PATTERN.search(text[max(0, m.start() - 40):m.end() + 40]))
            approach = ("ViV-TAVR" if is_viv else "TAVR") if tavr else \
                       ("ViV-SAVR" if is_viv else "SAVR")
            out.append(dict(approach=approach, is_viv=is_viv, valve_model=None,
                             valve_size_mm=None, service_date=date["year"],
                             date_precision=date["precision"], source="narrative_mention"))
    return out


def build_patient_timelines(notes_df):
    """
    notes_df: DataFrame with columns Profile Key, Type, Notes, Service Date.
    Returns dict: patient_id -> {
        'events': [ {service_date, approach, is_viv, valve_model, valve_size_mm, row_index} ... ] sorted by year,
        'index_implant_year': int or None,
        'index_implant_source': 'op_report_service_date' | 'undated',
        'index_valve_model': str or None,
        'index_valve_size_mm': int or None,
    }
    """
    timelines = {}
    for pid, sub in notes_df.groupby("Profile Key"):
        events = []
        for idx, row in sub.iterrows():
            ev = classify_note_as_implant_event(row["Type"], str(row["Notes"]))
            if ev:
                ev["service_date"] = int(row["Service Date"])
                ev["row_index"] = idx
                events.append(ev)
        used_narrative_fallback = False
        if not events:
            # No Operative Report/Procedures note in this extract for this
            # patient (observed in this corpus: a patient with only Progress
            # Notes surviving, which narrate the implant as history, e.g.
            # "s/p TAVR <year>"). Fall back to a narrative scan across every
            # note type for this patient.
            for idx, row in sub.iterrows():
                for ev in narrative_implant_mentions(str(row["Notes"])):
                    ev["row_index"] = idx
                    events.append(ev)
                    used_narrative_fallback = True
        events.sort(key=lambda e: (e["service_date"], e["is_viv"]))
        index_year, index_model, index_size, source, index_approach = None, None, None, "undated", None
        # index implant = earliest non-ViV event; if none, fall back to the
        # earliest event overall (patient enters the cohort already having
        # had ViV, or the index op report isn't in this extract).
        non_viv = [e for e in events if not e["is_viv"]]
        anchor = non_viv[0] if non_viv else (events[0] if events else None)
        if anchor:
            index_year = anchor["service_date"]
            index_model = anchor.get("valve_model")
            index_size = anchor.get("valve_size_mm")
            index_approach = anchor.get("approach")
            source = "narrative_mention" if used_narrative_fallback else "op_report_service_date"
        timelines[pid] = dict(
            events=events,
            index_implant_year=index_year,
            index_implant_source=source,
            index_valve_model=index_model,
            index_valve_size_mm=index_size,
            index_approach=index_approach,
        )
    return timelines


def years_since_index(service_date: int, index_implant_year):
    if index_implant_year is None:
        return None
    return service_date - index_implant_year
