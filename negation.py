"""
Lightweight negation handling for the clinical-note NLP layer.

Two mechanisms, because the notes mix two very different sources of false
positives (see architecture doc section 1, "Τα template πεδία παράγουν
ψευδώς θετικά"):

1. Structured "Field: Value" template lines (HVI-style op reports), where a
   positive keyword match ("Reoperation", "Valve in Valve") is immediately
   followed by a templated negative value ("No", "No previous surgeries").
2. Free-text negation cues preceding or following a clinical term
   ("denies", "no evidence of", "without", "ruled out", "negative for").

This is intentionally a simple, auditable cue-list negation detector (not a
statistical NegEx/ConText implementation) — appropriate for a ~215-note
corpus where every positive is going to be manually reviewed by the
physician anyway (Level 1 validation). Precision on the negatives is the
priority: we would rather under-flag than let a "No previous surgeries"
line masquerade as a reoperation event.
"""
import re

PRE_NEGATION_CUES = [
    r"\bno\b", r"\bno evidence of\b", r"\bwithout\b", r"\bw/o(?:ut)?\b",
    r"\bdenies?\b", r"\bdenied\b", r"\bnegative for\b", r"\bruled out\b",
    r"\brule out\b", r"\bnot\b", r"\babsent\b", r"\bfree of\b", r"\bnone\b",
]
POST_NEGATION_CUES = [
    r"\bwas ruled out\b", r"\bwas negative\b", r"\bnot (?:present|seen|identified|noted|found)\b",
    r"\bdenied\b",
]

_PRE_RE = re.compile("|".join(PRE_NEGATION_CUES), re.IGNORECASE)
_POST_RE = re.compile("|".join(POST_NEGATION_CUES), re.IGNORECASE)

# Values that make a "Field: <value>" template line a negative finding.
FIELD_NEGATIVE_VALUE = re.compile(
    r"^\s*(no\b.*|none\b.*|not\b.*|n/?a\b.*|negative\b.*|denied\b.*)?\s*$",
    re.IGNORECASE,
)
FIELD_POSITIVE_STUB = re.compile(r"^\s*(yes)\s*$", re.IGNORECASE)


def is_negated(text: str, start: int, end: int, window_chars: int = 40) -> bool:
    """Free-text negation check around a span [start:end) in `text`."""
    pre_lo = max(0, start - window_chars)
    pre_window = text[pre_lo:start]
    if _PRE_RE.search(pre_window[-window_chars:]):
        # guard against crossing a sentence boundary — a period between the
        # cue and the term usually means they're unrelated.
        segment = pre_window[-window_chars:]
        last_period = segment.rfind(".")
        cue_match = list(_PRE_RE.finditer(segment))
        if cue_match and (last_period == -1 or cue_match[-1].start() > last_period):
            return True
    post_hi = min(len(text), end + window_chars)
    post_window = text[end:post_hi]
    if _POST_RE.search(post_window):
        segment = post_window[:window_chars]
        first_period = segment.find(".")
        cue_match = _POST_RE.search(segment)
        if cue_match and (first_period == -1 or cue_match.start() < first_period):
            return True
    return False


def field_value_is_negative(value: str) -> bool:
    """For a parsed 'Field: value' pair, decide if the value is a negative
    template stub ('No', 'No previous surgeries', 'None', '') rather than a
    genuine positive finding."""
    value = (value or "").strip()
    if value == "":
        return True  # empty field, e.g. "Reoperation:" with nothing after it
    return bool(FIELD_NEGATIVE_VALUE.match(value))
