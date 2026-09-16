"""
Turns a note's raw text into (a) a dict of structured 'Field: Value' template
lines when present, and (b) explicit in-text date tokens (MM/YYYY, "Month
YYYY", parenthetical "(MM/YYYY)") that survive de-identification more often
than full dates — these give sub-year precision where available, which
matters because Service Date on the note itself is year-only (see
architecture doc section 3, "χρονικός περιορισμός").
"""
import re

# "Field Name: value" on its own line (allow the field name to span 1-4
# words, Title Case or ALL CAPS, as seen in both the HVI structured template
# and the narrative EPIC CARE headers like "PREOPERATIVE DIAGNOSES:").
FIELD_LINE = re.compile(
    r"^[ \t]*([A-Z][A-Za-z/ ]{1,40}?):[ \t]*(.*)$", re.MULTILINE
)

MONTHS = (
    "Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    "Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
DATE_TOKEN = re.compile(
    r"(?P<mmyyyy>\b(0?[1-9]|1[0-2])/(19|20)\d{2}\b)"
    r"|(?P<monyyyy>\b(?:" + MONTHS + r")\.?\s+(19|20)\d{2}\b)"
    r"|(?P<yyyy>\b(19|20)\d{2}\b)",
    re.IGNORECASE,
)

MONTH_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_field_lines(text: str) -> dict:
    """Return {field_name_lower: value_text} for template-style lines. Later
    duplicate field names overwrite earlier ones (last wins), which is fine
    for our purposes — the fields we key off of (Reoperation, Valve in
    Valve, Tissue Implant Type, Implant Size) appear at most once per note in
    this corpus."""
    fields = {}
    for m in FIELD_LINE.finditer(text):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        fields[key] = val
    return fields


def extract_date_tokens(text: str):
    """Return a list of {year, month(optional), start, end, matched_text}
    for every explicit date-like token found in free text. `month` is None
    for bare-year tokens."""
    out = []
    for m in DATE_TOKEN.finditer(text):
        if m.group("mmyyyy"):
            mm, yyyy = m.group("mmyyyy").split("/")
            out.append(dict(year=int(yyyy), month=int(mm), start=m.start(), end=m.end(),
                             matched_text=m.group(), precision="month"))
        elif m.group("monyyyy"):
            token = m.group("monyyyy")
            mon_str = re.match(r"[A-Za-z]+", token).group().lower()[:3]
            if mon_str == "sep" and token.lower().startswith("sept"):
                mon_str = "sept"
            year = int(re.search(r"(19|20)\d{2}", token).group())
            out.append(dict(year=year, month=MONTH_TO_NUM.get(mon_str), start=m.start(),
                             end=m.end(), matched_text=token, precision="month"))
        elif m.group("yyyy"):
            out.append(dict(year=int(m.group("yyyy")), month=None, start=m.start(),
                             end=m.end(), matched_text=m.group(), precision="year"))
    return out


def nearest_date_year(text: str, pos: int, window: int = 80):
    """Closest explicit date token (of any precision) to a position, used to
    time-stamp an individual clinical finding (e.g. a gradient value) more
    precisely than the note's own Service Date when the note narrates
    multiple visits (a common pattern in the Progress Notes here)."""
    tokens = extract_date_tokens(text)
    best, best_dist = None, None
    for t in tokens:
        mid = (t["start"] + t["end"]) / 2
        dist = abs(mid - pos)
        if dist <= window and (best_dist is None or dist < best_dist):
            best, best_dist = t, dist
    return best
