"""
Valve model dictionary and manufacturer/family lookup for the SVD phenotyping
engine (Dyania Docathon, Component 1).

Design note: several model names collide with unrelated vocabulary that is
common in these notes ("Epic" = the EHR system in "reviewed in Epic" / "EPIC
CARE OPERATIVE REPORT" header boilerplate, and "Epicardial" pacing wires).
Every dictionary entry below therefore carries a `require_context` regex that
must be found within CONTEXT_WINDOW characters of the match, and a
`block_context` regex that, if found overlapping the match window, vetoes it.
"""
import re

CONTEXT_WINDOW = 45  # chars each side to search for supporting/blocking context

# family: informs Bayesian AFT priors (Component 1, Head A) — early-SVD
# families (e.g. Trifecta) get an informative prior distinct from
# low-SVD-incidence families (e.g. mechanical-adjacent pericardial designs
# like Inspiris/Magna). This dictionary only *tags* the family; the prior
# values themselves belong in the modelling notebook, not here.
VALVE_MODELS = [
    # (canonical_name, family, regex, require_context, block_context)
    dict(name="Trifecta GT", family="Trifecta", pattern=r"\bTrifecta\s*GT\b"),
    dict(name="Trifecta", family="Trifecta", pattern=r"\bTrifecta\b"),
    dict(name="Perimount Magna Ease", family="Magna", pattern=r"\bMagna\s*Ease\b"),
    dict(name="Magna", family="Magna", pattern=r"\bMagna\b(?!\s*Ease)"),
    dict(name="Perimount", family="Perimount", pattern=r"\bPerimount\b"),
    dict(name="Inspiris Resilia", family="Inspiris", pattern=r"\bInspiris(?:\s*Resilia)?\b"),
    dict(name="Sapien S3", family="Sapien", pattern=r"\bSapien\s*S?3\b"),
    dict(name="Sapien", family="Sapien", pattern=r"\bSapien\b(?!\s*S?3)"),
    dict(name="Evolut FX", family="Evolut", pattern=r"\bEvolut\s*FX\b"),
    dict(name="Evolut PRO", family="Evolut", pattern=r"\bEvolut\s*PRO\+?\b"),
    dict(name="Evolut R", family="Evolut", pattern=r"\bEvolut\s*R\b"),
    dict(name="Evolut", family="Evolut", pattern=r"\bEvolut\b"),
    dict(
        name="Epic", family="Epic", pattern=r"\bEpic\b",
        require_context=r"valve|prosth|bioprosth|AVR|aortic",
        block_context=r"EPIC\s+CARE|reviewed in Epic|Epicardial|through Epic|Epic\s+(?:or|Mail|system)",
    ),
    dict(name="Biocor", family="Biocor", pattern=r"\bBiocor\b"),
    dict(name="Freestyle", family="Freestyle", pattern=r"\bFreestyle\b"),
    dict(name="Mosaic", family="Mosaic", pattern=r"\bMosaic\b"),
    dict(name="Hancock", family="Hancock", pattern=r"\bHancock\b(?:\s*II)?"),
    dict(name="Mitroflow", family="Mitroflow", pattern=r"\bMitroflow\b"),
    dict(name="Portico", family="Portico", pattern=r"\bPortico\b"),
    dict(name="Lotus", family="Lotus", pattern=r"\bLotus\b",
         require_context=r"valve|TAVR|transcatheter"),
    dict(name="Acurate neo", family="Acurate", pattern=r"\bAcurate\s*(?:neo2?)?\b"),
    dict(name="Centera", family="Centera", pattern=r"\bCentera\b"),
]

for _v in VALVE_MODELS:
    _v.setdefault("require_context", None)
    _v.setdefault("block_context", None)
    _v["compiled"] = re.compile(_v["pattern"], re.IGNORECASE)
    _v["compiled_require"] = re.compile(_v["require_context"], re.IGNORECASE) if _v["require_context"] else None
    _v["compiled_block"] = re.compile(_v["block_context"], re.IGNORECASE) if _v["block_context"] else None

# Families with a documented tendency toward earlier/more frequent SVD in the
# TVT/structural literature (used only to *tag* records; see study_protocol.md
# for citations backing the informative-prior claim before this ships).
EARLY_SVD_FAMILIES = {"Trifecta", "Mitroflow"}

SIZE_PATTERN = re.compile(r"(?<!\w)#?\s?(\d{2})\s?-?\s?mm\b", re.IGNORECASE)
# Weaker signal: a bare "#NN" (e.g. "Biocor prosthesis #27") with no "mm"
# unit. Only trusted very close to a valve-name mention (see
# nearest_size_mm) since "#NN" alone could be a clinic/ID number elsewhere
# in the note.
BARE_SIZE_PATTERN = re.compile(r"#\s?(\d{2})\b")


def find_valve_mentions(text: str):
    """Return list of dicts: {name, family, start, end, matched_text} for every
    valve-model mention in `text` that survives the context guard."""
    out = []
    for v in VALVE_MODELS:
        for m in v["compiled"].finditer(text):
            lo = max(0, m.start() - CONTEXT_WINDOW)
            hi = min(len(text), m.end() + CONTEXT_WINDOW)
            window = text[lo:hi]
            if v["compiled_block"] and v["compiled_block"].search(window):
                continue
            if v["compiled_require"] and not v["compiled_require"].search(window):
                continue
            out.append(dict(
                name=v["name"], family=v["family"],
                start=m.start(), end=m.end(), matched_text=m.group(),
            ))
    out.sort(key=lambda d: d["start"])
    return out


def nearest_size_mm(text: str, pos: int, window: int = 60, bare_window: int = 20):
    """Find the closest `NN mm` size token within `window` chars of position
    `pos`, falling back to a bare `#NN` token within the tighter
    `bare_window` if no `mm`-qualified size is nearby. Returns int or None."""
    lo, hi = max(0, pos - window), min(len(text), pos + window)
    best, best_dist = None, None
    for m in SIZE_PATTERN.finditer(text[lo:hi]):
        abs_pos = lo + m.start()
        dist = abs(abs_pos - pos)
        if best_dist is None or dist < best_dist:
            best, best_dist = int(m.group(1)), dist
    if best is not None:
        return best
    lo2, hi2 = max(0, pos - bare_window), min(len(text), pos + bare_window)
    for m in BARE_SIZE_PATTERN.finditer(text[lo2:hi2]):
        abs_pos = lo2 + m.start()
        dist = abs(abs_pos - pos)
        if best_dist is None or dist < best_dist:
            best, best_dist = int(m.group(1)), dist
    return best
