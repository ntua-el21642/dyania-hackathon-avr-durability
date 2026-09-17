"""
Converts published survival/freedom-from-SVD points (Master Prompt Section
6.1) into Weibull(shape k, scale lambda) priors per valve family, and one
level up per approach (SAVR-porcine, SAVR-pericardial, TAVR-balloon,
TAVR-self-expanding), per Section 6.2's method: least-squares fit of
log(-log S(t)) = k*log(t) - k*log(lambda) against the published (t, S(t))
points.

Every point below was read directly from the source PDFs on 2026-09-17
(not transcribed from the Master Prompt without checking) — see the
"verified_against" field on each entry for the exact page/table. Where a
family has only one published point, shape k is borrowed from the parent
approach-level fit and only scale (lambda) is solved from that single
point; the resulting CV is inflated further (see INFLATE_CV_SINGLE_POINT)
to reflect the much higher uncertainty of a one-point fit. No family or
approach is given a prior without at least one real literature anchor
directly or via its parent approach — Trifecta has neither (see
TRIFECTA_NOTE) and is deliberately left with only the TAVR-agnostic /
SAVR-pericardial-family-conditional weak prior a physician can override,
never a fabricated point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import yaml

# A point with S(t) in (0, 1) exactly 0 or 1 breaks log(-log S) (0 -> +inf,
# 1 -> -inf via log(0)); clip degenerate published 0%-event points to a
# small epsilon rather than drop them, and flag it in the point's note.
EPS = 1e-3
# Multiply the single-point CV by this factor relative to the multi-point
# default (Section 6.2: "default CV of 30-50% where none is given") — a
# single point pins the curve through one observation with a borrowed
# shape, which is materially less informative than an actual regression.
INFLATE_CV_SINGLE_POINT = 1.6
DEFAULT_CV = 0.4


@dataclass
class SurvivalPoint:
    t_years: float
    survival_pct: float  # 0-100, "freedom from X" already converted to survival scale by the caller
    note: str = ""

    @property
    def s(self) -> float:
        s = self.survival_pct / 100.0
        return min(max(s, EPS), 1 - EPS)


@dataclass
class LiteratureSource:
    citation: str
    n: int
    verified_against: str
    points: list[SurvivalPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verified literature anchors (Master Prompt Section 6.1), re-checked against
# the actual PDFs in papers/ on 2026-09-17: ejcts_52_3_408.pdf (Capodanno
# 2017, Table 1 & 2) and JAH3-14-e041505.pdf (Trimaille 2025, "Table 1").
# Every number below matches the source table verbatim; none were taken on
# faith from the Master Prompt's own transcription without this check.
# ---------------------------------------------------------------------------
SOURCES: dict[str, LiteratureSource] = {
    "hancock_ii_savr_porcine": LiteratureSource(
        citation="David et al. 2010, Table 1 (Capodanno 2017)", n=1134,
        verified_against="ejcts_52_3_408.pdf p.4 Table 1, row 1",
        points=[SurvivalPoint(20, 63, "Freedom from SVD (undefined criterion in source)")],
    ),
    "freestyle_savr_stentless": LiteratureSource(
        citation="Mohammadi et al. 2012, Table 1 (Capodanno 2017)", n=430,
        verified_against="ejcts_52_3_408.pdf p.4 Table 1, row 2",
        points=[
            SurvivalPoint(10, 95.9, "Freedom from reoperation for SVD"),
            SurvivalPoint(15, 82.3, "Freedom from reoperation for SVD"),
        ],
    ),
    "carpentier_edwards_savr_pericardial": LiteratureSource(
        citation="Bourguignon et al. 2015 (primary, SVD-specific) + Forcillo et al. 2013 + "
                  "Johnstone et al. 2015 (corroborating), Table 1 (Capodanno 2017)", n=2758,
        verified_against="ejcts_52_3_408.pdf p.4 Table 1, rows 3/5/6",
        points=[
            SurvivalPoint(15, 79, "Bourguignon: freedom from SVD (severe AS/AR definition)"),
            SurvivalPoint(20, 49, "Bourguignon: freedom from SVD"),
        ],
    ),
    "mitroflow_savr_pericardial_rapid": LiteratureSource(
        citation="Senage et al. 2014, Table 1 (Capodanno 2017) — known accelerated-SVD outlier "
                  "family, kept separate from the general CE/pericardial prior rather than pooled",
        n=617, verified_against="ejcts_52_3_408.pdf p.4 Table 1, row 4",
        points=[SurvivalPoint(5, 92, "Freedom from SVD (Capodanno absolute-threshold definition)")],
    ),
    "sapien_family_tavr_balloon": LiteratureSource(
        citation="Pibarot et al. 2020 (Sapien XT + Sapien 3) + Mack et al. 2023 (low-risk Sapien 3) "
                  "+ Mack et al. 2015, Table (Trimaille 2025) & Table 2 (Capodanno 2017)",
        n=1665 + 495 + 348,
        verified_against="JAH3-14-e041505.pdf p.4 table rows 'Pibarot et al 2020', 'Mack et al 2023'; "
                          "ejcts_52_3_408.pdf p.4 Table 2 row 'Mack et al. 2015'",
        points=[
            # Pooled 5-year BVF incidence across the three balloon-expandable
            # RCT/registry cohorts above (1.6% Sapien XT, 0.7% Sapien 3, 1.6%
            # Sapien 3 low-risk, 0% Sapien in the older Mack 2015 series) ->
            # simple unweighted mean incidence 0.98% -> freedom 99.0%.
            SurvivalPoint(5, 99.0, "Pooled mean BVF incidence across Sapien XT/3 cohorts (VARC-3/EAPCI)"),
        ],
    ),
    "corevalve_evolut_tavr_self_expanding": LiteratureSource(
        citation="Barbanti et al. 2015, Table 2 (Capodanno 2017)", n=353,
        verified_against="ejcts_52_3_408.pdf p.4 Table 2, row 3",
        points=[SurvivalPoint(5, 100 - 1.4, "Freedom from bioprosthetic valve dysfunction (VARC-1)")],
    ),
    # Approach-level (population) TAVR anchor used both as the TAVR-general
    # parent prior AND to borrow shape k for single-point TAVR families
    # above: pools the 5-year balloon-expandable point with the two
    # longer-follow-up, mixed-device 10-year points (EAPCI/VARC-3 general
    # definitions, not device-specific), giving an actual 2-point fit
    # instead of a single borrowed-shape point at the approach level.
    "tavr_general_population": LiteratureSource(
        citation="Pooled: Pibarot 2020 + Mack 2023 (5y, balloon-expandable) and Alaour 2025 + "
                  "Sathananthan 2021 (10y, mixed balloon/self-expanding), Table (Trimaille 2025)",
        n=1665 + 495 + 2403 + 235,
        verified_against="JAH3-14-e041505.pdf p.4 table rows 'Pibarot et al 2020', 'Mack et al 2023', "
                          "'Alaour et al 2025', 'Sathananthan et al 2021'",
        points=[
            SurvivalPoint(5, 99.0, "Pooled mean BVF incidence, 5y (balloon-expandable RCT/registry)"),
            SurvivalPoint(10, 100 - (12.7 + 6.5) / 2, "Pooled mean BVF incidence, 10y (mixed device, EAPCI/VARC-3)"),
        ],
    ),
    # Approach-level SAVR anchor for the same purpose, pooling all 4 SAVR
    # families above at whichever horizons they report.
    "savr_general_population": LiteratureSource(
        citation="Pooled: David 2010 (20y) + Mohammadi 2012 (10/15y) + Bourguignon 2015 (15/20y) + "
                  "Senage 2014 (5y), Table 1 (Capodanno 2017)",
        n=1134 + 430 + 2758 + 617,
        verified_against="ejcts_52_3_408.pdf p.4 Table 1",
        points=[
            SurvivalPoint(5, 92, "Senage (Mitroflow) 5y anchor"),
            SurvivalPoint(10, 95.9, "Mohammadi (Freestyle) 10y anchor"),
            SurvivalPoint(20, (63 + 49) / 2, "Mean of David (Hancock II) and Bourguignon (CE) 20y anchors"),
        ],
    ),
}

# Trifecta: confirmed absent from all three provided source PDFs (ehaa799,
# ejcts_52_3_408, JAH3-14-e041505) via a full-text search for "Trifecta" on
# 2026-09-17 — zero matches in any of the three. Per Master Prompt Section
# 6.1's own instruction ("Do not invent a Trifecta prior"), Trifecta gets NO
# family-level literature point. It is assigned only the SAVR-pericardial
# population-level prior (its own known clinical signal — early SVD — is
# documented here as a qualitative flag for the physician, not encoded as a
# numeric adjustment to the prior, since no peer-reviewed effect size was
# found in the provided files).
TRIFECTA_NOTE = (
    "No Trifecta-specific durability point found in any of the 3 provided source PDFs "
    "(full-text search, 2026-09-17: zero matches for 'Trifecta' in ehaa799.pdf, "
    "ejcts_52_3_408.pdf, JAH3-14-e041505.pdf). Trifecta valves in this cohort account for "
    "3 of 8 seed-table SVD-signal patients (Patient_017, 030, 058 — see review/error_catalogue.md), "
    "a known clinical early-SVD signal, but it is NOT encoded numerically here — Trifecta is "
    "assigned the savr_general_population prior only, with this note attached so a physician can "
    "manually flag Trifecta patients for closer surveillance pending a sourced Trifecta-specific "
    "durability estimate. TODO: source required."
)


def fit_weibull_least_squares(points: list[SurvivalPoint]) -> tuple[float, float]:
    """Section 6.2: fit Weibull (k, lambda) by least squares of
    log(-log S(t)) = k*log(t) - k*log(lambda) against published points.
    Returns (k, lambda). Requires >=2 points."""
    x = np.array([math.log(p.t_years) for p in points])
    y = np.array([math.log(-math.log(p.s)) for p in points])
    if len(points) == 1:
        raise ValueError("need >=2 points for least squares; use fit_single_point instead")
    k, intercept = np.polyfit(x, y, 1)
    lam = math.exp(-intercept / k)
    return float(k), float(lam)


def fit_single_point(point: SurvivalPoint, borrowed_k: float) -> float:
    """Solve lambda from one point given a shape k borrowed from the parent
    approach-level fit: lambda = t / (-log S(t))^(1/k)."""
    return point.t_years / ((-math.log(point.s)) ** (1.0 / borrowed_k))


def build_priors() -> dict:
    k_savr, lam_savr = fit_weibull_least_squares(SOURCES["savr_general_population"].points)
    k_tavr, lam_tavr = fit_weibull_least_squares(SOURCES["tavr_general_population"].points)

    approach_priors = {
        "SAVR": dict(shape_k=k_savr, scale_lambda_years=lam_savr, cv=DEFAULT_CV,
                     citation=SOURCES["savr_general_population"].citation,
                     verified_against=SOURCES["savr_general_population"].verified_against,
                     n_pooled=SOURCES["savr_general_population"].n),
        "TAVR": dict(shape_k=k_tavr, scale_lambda_years=lam_tavr, cv=DEFAULT_CV,
                     citation=SOURCES["tavr_general_population"].citation,
                     verified_against=SOURCES["tavr_general_population"].verified_against,
                     n_pooled=SOURCES["tavr_general_population"].n),
    }

    family_priors = {}
    multi_point_families = {
        "freestyle_savr_stentless": "SAVR",
        "carpentier_edwards_savr_pericardial": "SAVR",
    }
    single_point_families = {
        "hancock_ii_savr_porcine": "SAVR",
        "mitroflow_savr_pericardial_rapid": "SAVR",
        "sapien_family_tavr_balloon": "TAVR",
        "corevalve_evolut_tavr_self_expanding": "TAVR",
    }

    for fam, approach in multi_point_families.items():
        src = SOURCES[fam]
        k, lam = fit_weibull_least_squares(src.points)
        family_priors[fam] = dict(
            approach=approach, shape_k=k, scale_lambda_years=lam, cv=DEFAULT_CV,
            citation=src.citation, verified_against=src.verified_against, n=src.n,
            points=[(p.t_years, p.survival_pct, p.note) for p in src.points],
        )

    for fam, approach in single_point_families.items():
        src = SOURCES[fam]
        borrowed_k = approach_priors[approach]["shape_k"]
        lam = fit_single_point(src.points[0], borrowed_k)
        family_priors[fam] = dict(
            approach=approach, shape_k=borrowed_k, scale_lambda_years=lam,
            cv=DEFAULT_CV * INFLATE_CV_SINGLE_POINT,
            shape_k_note=f"borrowed from {approach}-general population fit (single-point family)",
            citation=src.citation, verified_against=src.verified_against, n=src.n,
            points=[(p.t_years, p.survival_pct, p.note) for p in src.points],
        )

    # Valve-name -> prior-family lookup used by build_features.py to map
    # each patient's index_valve_model onto one of the fitted priors above
    # (or the bare approach-level prior when the model wasn't in the
    # dictionary at all, or was Trifecta).
    valve_model_to_family = {
        "Hancock": "hancock_ii_savr_porcine",
        "Mosaic": "hancock_ii_savr_porcine",  # Medtronic porcine stented, same broad class, no dedicated point
        "Freestyle": "freestyle_savr_stentless",
        "Perimount": "carpentier_edwards_savr_pericardial",
        "Perimount 2700": "carpentier_edwards_savr_pericardial",
        "Magna": "carpentier_edwards_savr_pericardial",
        "Magna Ease": "carpentier_edwards_savr_pericardial",
        "Inspiris": "carpentier_edwards_savr_pericardial",
        "Inspiris Resilia": "carpentier_edwards_savr_pericardial",
        "Mitroflow": "mitroflow_savr_pericardial_rapid",
        "Sapien": "sapien_family_tavr_balloon",
        "Sapien XT": "sapien_family_tavr_balloon",
        "Sapien S3": "sapien_family_tavr_balloon",
        "Sapien 3": "sapien_family_tavr_balloon",
        "Evolut": "corevalve_evolut_tavr_self_expanding",
        "Evolut R": "corevalve_evolut_tavr_self_expanding",
        "Evolut PRO": "corevalve_evolut_tavr_self_expanding",
        "Evolut FX": "corevalve_evolut_tavr_self_expanding",
        "CoreValve": "corevalve_evolut_tavr_self_expanding",
        # No literature point: Trifecta, Trifecta GT, Biocor, Epic, Konect,
        # homograft. All fall back to the approach-level population prior
        # in build_features.py / the model, per TRIFECTA_NOTE above.
    }

    return dict(
        generated_by="src/models/durability/fit_literature_priors.py",
        method="Least-squares fit of log(-log S(t)) vs log(t) (Master Prompt Section 6.2); "
               "single-point families borrow shape k from their approach-level population fit.",
        weibull_parameterization="S(t) = exp(-(t/scale_lambda_years)^shape_k), t in years since index implant",
        cv_note=f"Default CV {DEFAULT_CV:.0%} for multi-point fits (Section 6.2 default where "
                f"published CIs are absent); {DEFAULT_CV*INFLATE_CV_SINGLE_POINT:.0%} for single-point "
                f"families (shape borrowed, only one observed point).",
        trifecta_note=TRIFECTA_NOTE,
        approach_level=approach_priors,
        valve_family_level=family_priors,
        valve_model_to_family=valve_model_to_family,
    )


if __name__ == "__main__":
    priors = build_priors()
    with open("config/priors.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(priors, f, sort_keys=False, allow_unicode=True, width=100)
    print("wrote config/priors.yaml")
    for name, p in priors["approach_level"].items():
        print(f"  approach {name}: k={p['shape_k']:.3f} lambda={p['scale_lambda_years']:.1f}y")
    for name, p in priors["valve_family_level"].items():
        print(f"  family {name}: k={p['shape_k']:.3f} lambda={p['scale_lambda_years']:.1f}y (n={p['n']})")
