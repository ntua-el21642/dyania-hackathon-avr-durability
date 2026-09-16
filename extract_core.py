"""
Per-note clinical finding extraction: gradients/EOA/DVI, LVEF, PVL/AR grade,
redo/reoperation, ViV, and the exclusion/definite-label trigger terms
(endocarditis, thrombosis, explicit SVD/BVF language).

Every extracted finding carries the character span it came from (for the
physician review sheet — architecture doc "ανάλυση σφαλμάτων" / Level 1
validation needs the evidence sentence, not just the parsed value) and an
`is_post_implant` flag once combined with the note's own years-since-index
in pipeline.py.
"""
import re
from negation import is_negated, field_value_is_negative
from sectioning import parse_field_lines, nearest_date_year

# ---------------------------------------------------------------------------
# Hemodynamics: peak/mean gradient, EOA, DVI/dimensionless index, peak velocity
# ---------------------------------------------------------------------------
# Three template families observed in this corpus (see architecture notes):
#   (a) "The peak gradient is X mmHg, the mean gradient is Y mmHg and the
#        dimensionless valve index is Z."   <- dominant TTE-report template
#   (b) "Peak/mean gradient X/Y mmHg" / "Pk/Mn gradient X/Y mmHg" / "peak and
#        mean gradients were X/Y mm[Hg] respectively"
#   (c) "AV mean gradient X mmHg" (mean alone, no paired peak in the sentence)
GRADIENT_PATTERNS = [
    re.compile(
        r"peak gradient is\s*(?P<peak>\d{1,3}(?:\.\d)?)\s*mmHg.{0,40}?"
        r"mean gradient is\s*(?P<mean>\d{1,3}(?:\.\d)?)\s*mmHg"
        r"(?:.{0,40}?dimensionless (?:valve )?index is\s*(?P<dvi>\d\.\d{1,2}))?",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:peak\s*/\s*mean|pk\s*/\s*mn)\s*gradients?\s*(?:of|were|is)?\s*"
        r"(?P<peak>\d{1,3}(?:\.\d)?)\s*/\s*(?P<mean>\d{1,3}(?:\.\d)?)\s*mm(?:Hg)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"peak and mean gradients? (?:were|are|of)\s*"
        r"(?P<peak>\d{1,3}(?:\.\d)?)\s*(?:and|/)\s*(?P<mean>\d{1,3}(?:\.\d)?)\s*mm",
        re.IGNORECASE,
    ),
    re.compile(
        r"AV mean grad(?:ient)?\s*(?:is|of)?\s*(?P<mean>\d{1,3}(?:\.\d)?)\s*mmHg",
        re.IGNORECASE,
    ),
    re.compile(
        r"mean grad(?:ient)?\s*(?P<mean>\d{1,3}(?:\.\d)?)\s*mmHg",
        re.IGNORECASE,
    ),
]

DVI_STANDALONE = re.compile(
    r"(?:dimensionless (?:valve )?index|DVI)\s*(?:is|of)?\s*(?P<dvi>\d\.\d{1,2})",
    re.IGNORECASE,
)
EOA_PATTERN = re.compile(
    r"(?:aortic )?valve area (?:is|of)?\s*(?P<eoa>\d\.\d{1,2})\s*cm",
    re.IGNORECASE,
)
LVEF_PATTERN = re.compile(
    r"\bLVEF\b[\s:]*(?:of|is)?\s*(?P<lvef>\d{1,2})\s*%|"
    r"\bEF\b[\s:]*(?:of|is)?\s*(?P<lvef2>\d{1,2})\s*%",
    re.IGNORECASE,
)
AR_GRADE_PATTERN = re.compile(
    r"(?P<grade>trace|mild(?:-to-moderate)?|moderate(?:-to-severe)?|severe)\s*"
    r"(?:para)?(?:valvular)?\s*(?:aortic )?(?:regurgitation|\bAR\b|\bAI\b|leak)",
    re.IGNORECASE,
)
AR_GRADE_ORDER = {"none": 0, "trace": 1, "mild": 1, "mild-to-moderate": 2,
                   "moderate": 2, "moderate-to-severe": 3, "severe": 3}


VALVE_IDENTITY = re.compile(
    r"(?P<aortic>\baortic valve\b|\bAV\b|\baortic\b)|"
    r"(?P<other>\bmitral valve\b|\bMV\b|\bmitral\b|\btricuspid valve\b|\bTV\b|"
    r"\btricuspid\b|\bpulmonic\b|\bpulmonary valve\b|\bPV\b)",
    re.IGNORECASE,
)


def nearest_valve_context(text: str, pos: int, window: int = 400):
    """Which valve (aortic vs. other) the text at `pos` is most likely
    talking about, based on the closest preceding valve-identity keyword.
    Echo reports here narrate one valve at a time in prose (observed in this
    corpus: a mitral-valve paragraph whose gradient sentence doesn't repeat
    'mitral' again until a trailing prior-gradients summary later in the
    same paragraph), so 'closest keyword wins' is the right heuristic rather
    than a fixed short window. Returns 'aortic', 'other', or None if no
    keyword found in range."""
    lo = max(0, pos - window)
    segment = text[lo:pos]
    last = None
    for m in VALVE_IDENTITY.finditer(segment):
        last = m
    if last is None:
        return None
    return "aortic" if last.group("aortic") else "other"


def extract_gradients(text: str):
    """Return list of {peak, mean, dvi, start, end, year} — one per AORTIC
    valve gradient statement found, deduplicated by (peak, mean) so a value
    repeated near-verbatim in a copy-forwarded note isn't double counted.
    Skips a match if a mitral/tricuspid/pulmonic marker sits immediately
    before it (e.g. a trailing prior-mitral-gradients summary line) — this
    corpus narrates all four valves with the same 'peak/mean gradient'
    phrasing, so the valve identity has to be disambiguated from the
    immediate left context rather than assumed aortic."""
    out, seen = [], set()
    for pat in GRADIENT_PATTERNS:
        for m in pat.finditer(text):
            gd = m.groupdict()
            mean = gd.get("mean")
            if mean is None:
                continue
            if nearest_valve_context(text, m.start()) == "other":
                continue
            peak = gd.get("peak")
            dvi = gd.get("dvi")
            key = (peak, mean)
            if key in seen:
                continue
            seen.add(key)
            if dvi is None:
                dvi_m = DVI_STANDALONE.search(text, m.end(), min(len(text), m.end() + 60))
                dvi = dvi_m.group("dvi") if dvi_m else None
            date = nearest_date_year(text, m.start(), window=100)
            out.append(dict(
                peak=float(peak) if peak else None,
                mean=float(mean),
                dvi=float(dvi) if dvi else None,
                start=m.start(), end=m.end(),
                context_year=date["year"] if date else None,
                context_year_precision=date["precision"] if date else None,
                matched_text=m.group(),
            ))
    return out


def extract_eoa(text: str):
    return [dict(eoa=float(m.group("eoa")), start=m.start(), end=m.end())
            for m in EOA_PATTERN.finditer(text)]


def extract_lvef(text: str):
    out = []
    for m in LVEF_PATTERN.finditer(text):
        val = m.group("lvef") or m.group("lvef2")
        if val:
            out.append(dict(lvef=int(val), start=m.start(), end=m.end()))
    return out


def extract_ar_grade(text: str):
    """Highest-severity AR/PVL grade mentioned, with negation filtering
    ('no aortic regurgitation', 'trivial AR' handled as trace)."""
    out = []
    for m in AR_GRADE_PATTERN.finditer(text):
        if is_negated(text, m.start(), m.end()):
            continue
        grade_raw = m.group("grade").lower()
        out.append(dict(grade=grade_raw, ordinal=AR_GRADE_ORDER.get(grade_raw, 1),
                         start=m.start(), end=m.end(), matched_text=m.group()))
    return out


# ---------------------------------------------------------------------------
# Redo / reoperation / ViV — structured field first, narrative fallback
# ---------------------------------------------------------------------------
REDO_NARRATIVE = re.compile(
    # Deliberately NOT ending on bare "surgery" (was too permissive: matched
    # "redo cardiac surgery" for a patient's unrelated redo-sternotomy — see
    # data_plan.md "known limitations"). The qualifying token must itself be
    # valve/root-specific. [\s\S]{0,80}? (not [^.]{0,60}?) so a "Redo.\nRoot
    # and AVR replacement..." list-item style entry — "Redo" as its own
    # sentence, the procedure named in the next one — still matches; this
    # was the second largest source of false negatives during development.
    r"\bredo\b[\s\S]{0,80}?(?:sternotomy|AVR|aortic valve|aortic root|root replacement|homograft)|"
    r"\bre-?operation\b|\breoperative\b",
    re.IGNORECASE,
)
# NOTE: deliberately NOT matching bare "prior cardiac surgery" / "previous
# surgery" — these are generic echo/imaging boilerplate ("septal motion
# abnormal consistent with prior cardiac surgery") that fires on the
# patient's own index operation and carries no reintervention information.
VIV_NARRATIVE = re.compile(
    r"valve[- ]in[- ]valve|\bViV\b|\bVIV[- ]?TAVR\b", re.IGNORECASE
)
# A "Plan: redo AVR" / "will undergo redo AVR" / "candidate for redo AVR"
# mention describes a FUTURE, not-yet-performed procedure — must not be
# counted as a completed reintervention (definite BVF Stage 2). Checked
# immediately before a redo/ViV narrative match.
PLANNED_FRAME = re.compile(
    r"\bplan(?:ned)?\b\s*:?|\bwill undergo\b|\brecommend(?:ed|ation)?\b|"
    r"\bcandidate for\b|\bscheduled for\b|\bto undergo\b",
    re.IGNORECASE,
)
ENDOCARDITIS_PATTERN = re.compile(r"\bendocarditis\b", re.IGNORECASE)
# "Endocarditis prophylaxis [therapy] for dental procedures" is generic
# preventive-care boilerplate present in most structural-heart follow-up
# notes; it is not a diagnosis and must not trigger the exclusion bucket.
ENDOCARDITIS_PROPHYLAXIS_BLOCK = re.compile(
    r"endocarditis\s+prophylaxis", re.IGNORECASE
)
THROMBOSIS_PATTERN = re.compile(r"\bthrombos[ei]s\b|\bthrombus\b|\bHALT\b", re.IGNORECASE)
# Paravalvular leak / PVL is non-structural (NSVD) per VARC-3 — a redo or
# "prosthetic valve regurgitation" trigger driven by PVL must route to the
# exclusion bucket, not to a definite SVD/BVF label.
PVL_PATTERN = re.compile(
    r"\bPVL\b|para-?valvular\s+(?:leak|regurgitation)", re.IGNORECASE
)
SVD_EXPLICIT_PATTERN = re.compile(
    r"structural valve deterioration|\bSVD\b(?!\w)|"
    r"prosthetic (?:aortic )?valve (?:failure|stenosis|regurgitation)|"
    r"bioprosthetic (?:AVR|valve) (?:stenosis|failure|deterioration)|"
    r"progressive (?:bioprosthetic )?(?:AVR|valve) stenosis|"
    r"valve dysfunction",
    re.IGNORECASE,
)
# Qualitative hemodynamic worsening with NO extractable number ("increased
# AVR gradients", "increasing gradients") — can't be VARC-3 staged (no
# value to threshold), but is a genuine "possible" signal, not nothing.
GRADIENT_TREND_PATTERN = re.compile(
    r"increas(?:ed|ing)\s+(?:AVR\s+|prosthetic\s+)?gradients?\b",
    re.IGNORECASE,
)


def valve_context_at(text: str, start: int, end: int, window: int = 250):
    """Bidirectional version of nearest_valve_context: the closest
    valve-identity keyword either before or after the span, whichever is
    nearer. Needed for generic phrases like "prosthetic valve dysfunction"
    where the disambiguating valve name (aortic vs. mitral) can follow
    rather than precede the match, observed in this corpus."""
    lo, hi = max(0, start - window), min(len(text), end + window)
    best_dist, best_family = None, None
    for m in VALVE_IDENTITY.finditer(text[lo:hi]):
        abs_start = lo + m.start()
        if abs_start < start:
            dist = start - (lo + m.end())
        elif abs_start >= end:
            dist = abs_start - end
        else:
            continue  # keyword overlaps the match itself; ignore
        family = "aortic" if m.group("aortic") else "other"
        if best_dist is None or dist < best_dist:
            best_dist, best_family = dist, family
    return best_family


def _structured_or_narrative(text: str, field_names, narrative_pattern):
    """Check a structured template field first (authoritative negation via
    field_value_is_negative); fall back to narrative regex + negation.py if
    the field isn't present in this note. Returns list of positive
    mention dicts."""
    fields = parse_field_lines(text)
    for fname in field_names:
        if fname in fields:
            if field_value_is_negative(fields[fname]):
                return []  # authoritative negative from the template — done
            return [dict(source="structured_field", field=fname,
                          value=fields[fname], start=None, end=None)]
    out = []
    for m in narrative_pattern.finditer(text):
        if is_negated(text, m.start(), m.end()):
            continue
        # "Plan: redo AVR" / "will undergo ViV TAVR" describes a FUTURE
        # procedure, not a completed one — must not count as a definite
        # reintervention (observed in this corpus: a pre-op risk assessment
        # recommending a not-yet-performed redo).
        left = text[max(0, m.start() - 40):m.start()]
        if PLANNED_FRAME.search(left):
            continue
        out.append(dict(source="narrative", start=m.start(), end=m.end(),
                         matched_text=m.group()))
    return out


PMH_FRAME = re.compile(
    r"past (?:medical|surgical) history|\bPMH\b|\bPSH\b", re.IGNORECASE
)


def extract_redo(text: str):
    hits = _structured_or_narrative(text, ["reoperation"], REDO_NARRATIVE)
    out = []
    for h in hits:
        if h.get("start") is not None:
            # A redo mention introduced by a PMH/PSH frame (observed in this
            # corpus: a past-medical-history line listing a prior redo
            # sternotomy and valve procedures) is restating a *known* prior
            # operation (very often the same
            # index implant already captured from its own Operative Report,
            # done via a redo-sternotomy approach because of unrelated
            # earlier surgery), not reporting a new reintervention. Without
            # a distinct date this can't be told apart from a genuine new
            # redo by keyword matching alone — this is exactly the class of
            # ambiguity the physician blind-review pass (Level 1) is for,
            # so we suppress it here rather than guess, and it should still
            # surface in the reviewer sheet's low-confidence queue upstream
            # if other evidence exists.
            left = text[max(0, h["start"] - 45):h["start"]]
            if PMH_FRAME.search(left):
                continue
        out.append(h)
    return out


def extract_viv(text: str):
    return _structured_or_narrative(text, ["valve in valve"], VIV_NARRATIVE)


IMPLICIT_NEW_TAVR = re.compile(
    r"\b(?:had|underwent|s/p)\s+(?:a\s+)?(?:successful\s+)?(?:TF[- ]|TA[- ])?TAVR\b",
    re.IGNORECASE,
)


def extract_implicit_new_tavr(text: str):
    """A bare 's/p TAVR' / 'underwent TAVR' mention with no explicit
    'valve-in-valve' wording (observed in this corpus: a patient whose note
    narrates developing prosthetic valve stenosis and then getting a TAVR,
    without ever using the words "valve-in-valve"). Only meaningful as a
    reintervention signal for a patient whose INDEX implant was surgical
    (SAVR) — a TAVR placed into an already-surgical prosthesis is a ViV by
    definition even when the note doesn't use that term. Caller
    (pipeline.py) is responsible for gating this on index approach; this
    function only does the text-level extraction + negation."""
    out = []
    for m in IMPLICIT_NEW_TAVR.finditer(text):
        if is_negated(text, m.start(), m.end()):
            continue
        out.append(dict(source="narrative_implicit_tavr", start=m.start(), end=m.end(),
                         matched_text=m.group()))
    return out


def extract_exclusions(text: str):
    """Endocarditis / thrombosis / PVL-driven dysfunction — these route to
    the 'exclusion / competing event' bucket (architecture doc section 3),
    never to a features row, per the leakage guard."""
    out = {}
    for name, pat in [("endocarditis", ENDOCARDITIS_PATTERN),
                       ("thrombosis_or_halt", THROMBOSIS_PATTERN),
                       ("pvl", PVL_PATTERN)]:
        hits = [m for m in pat.finditer(text) if not is_negated(text, m.start(), m.end())]
        if name == "endocarditis":
            hits = [m for m in hits if not ENDOCARDITIS_PROPHYLAXIS_BLOCK.search(
                text[max(0, m.start() - 15):m.end() + 15])]
        if hits:
            out[name] = [dict(start=m.start(), end=m.end(), matched_text=m.group())
                          for m in hits]
    return out


def extract_gradient_trend(text: str):
    """Qualitative hemodynamic worsening with no extractable numeric value
    ("increased AVR gradients", no mmHg figure given) — can't be VARC-3
    staged, but is a genuine 'possible' signal rather than nothing, and
    should surface for physician review rather than be silently dropped."""
    out = []
    for m in GRADIENT_TREND_PATTERN.finditer(text):
        if is_negated(text, m.start(), m.end()):
            continue
        out.append(dict(start=m.start(), end=m.end(), matched_text=m.group()))
    return out


MORPHOLOGY_PATTERN = re.compile(
    r"(?:leaflet\s+(?:thickening|calcification|restrict\w*|immobil\w*)|"
    r"restrict\w*\s+leaflet\s+motion|reduced\s+leaflet\s+(?:excursion|mobility)|"
    r"\bHALT\b|hypo-?attenuated leaflet)",
    re.IGNORECASE,
)


def extract_morphology(text: str):
    """Morphological-only findings (leaflet thickening/calcification, HALT,
    reduced leaflet mobility) with no accompanying hemodynamic change —
    the 'Possible' tier / candidate Stage 1 in the ground-truth hierarchy
    (architecture doc section 3)."""
    out = []
    for m in MORPHOLOGY_PATTERN.finditer(text):
        if is_negated(text, m.start(), m.end()):
            continue
        out.append(dict(start=m.start(), end=m.end(), matched_text=m.group()))
    return out


def extract_svd_explicit(text: str):
    """Direct textual SVD/BVF language — the 'Definite' tier of the ground
    truth hierarchy (architecture doc section 3). The generic phrasings in
    SVD_EXPLICIT_PATTERN ('prosthetic valve stenosis/failure', 'valve
    dysfunction') are also used verbatim for the mitral and tricuspid
    prostheses in this corpus, immediately followed by the disambiguating
    valve name (e.g. "...prosthetic valve dysfunction, mitral valve..."),
    so anything not containing an explicit 'aortic'/'AVR'/'AV' cue in the
    match itself is required to have aortic
    as its nearest valve-identity context, not mitral/tricuspid/pulmonic."""
    out = []
    for m in SVD_EXPLICIT_PATTERN.finditer(text):
        if is_negated(text, m.start(), m.end()):
            continue
        if not re.search(r"aortic|\bAVR\b|\bAV\b", m.group(), re.IGNORECASE):
            ctx = valve_context_at(text, m.start(), m.end())
            if ctx == "other":
                continue
        out.append(dict(start=m.start(), end=m.end(), matched_text=m.group()))
    return out


def sentence_window(text: str, start: int, end: int, pad: int = 120) -> str:
    """Evidence snippet for the reviewer sheet: the sentence(s) around a
    match, trimmed to whole-ish sentence boundaries where possible."""
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    snippet = text[lo:hi].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet
