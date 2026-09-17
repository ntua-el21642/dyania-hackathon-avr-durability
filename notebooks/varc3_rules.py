"""
Deterministic VARC-3 rule engine — Head B of the architecture (section 5):
"δεν είναι εκπαιδευμένο μοντέλο ... πλήρως ελέγξιμη από τον κλινικό." This
module takes the per-patient aggregated evidence produced by pipeline.py and
assigns:
  - HVD/SVD stage (0 / 1 / 2 / 3, with S/R/RS phenotype note at stage 2-3)
  - BVF stage (0 / 2 / 3) — Stage 1 (asymptomatic HVD) is not itself BVF;
    BVF requires reintervention (Stage 2) or valve-related death (Stage 3,
    not derivable from this notes-only extract — flagged, not fabricated)
  - confidence tier: definite / probable / possible / excluded / censored
  - a human-readable rationale string for the reviewer sheet

Thresholds
----------
VARC-3 Table 12/13 (primary): stage requires a *concurrent* EOA/DVI drop,
not gradient alone — this is what distinguishes true HVD from PPM/high-flow
states (architecture doc section 2). Because EOA/DVI/baseline echo are only
recoverable for a minority of notes here, the VARC-3-strict path is used
when we have both a gradient delta AND an EOA or DVI delta from a
recoverable pre-implant/early-postop baseline. Elsewhere we fall back to the
Capodanno 2017 absolute thresholds (no baseline required) and mark the
result 'probable', exactly as specified.
"""

# --- VARC-3 Table 13 (delta-based, requires a baseline echo) ---------------
VARC3_STAGE3_DELTA_MG = 20
VARC3_STAGE3_ABS_MG = 30
VARC3_STAGE3_EOA_DROP_CM2 = 0.6
VARC3_STAGE3_EOA_DROP_PCT = 0.50
VARC3_STAGE3_DVI_DROP_ABS = 0.2
VARC3_STAGE3_DVI_DROP_PCT = 0.40

VARC3_STAGE2_DELTA_MG = 10
VARC3_STAGE2_ABS_MG = 20
VARC3_STAGE2_EOA_DROP_CM2 = 0.3
VARC3_STAGE2_EOA_DROP_PCT = 0.25
VARC3_STAGE2_DVI_DROP_ABS = 0.1
VARC3_STAGE2_DVI_DROP_PCT = 0.20

# --- Capodanno 2017 absolute thresholds (no baseline required) -------------
CAPODANNO_SEVERE_MG = 40
CAPODANNO_MODERATE_MG = 20


def classify_hvd_stage_from_gradients(baseline_mg, followup_mg, baseline_eoa=None,
                                       followup_eoa=None, baseline_dvi=None,
                                       followup_dvi=None):
    """VARC-3-strict path: requires a baseline. Returns (stage, rationale) or
    (None, reason) if the concurrent EOA/DVI requirement isn't met (i.e. the
    gradient rose but nothing suggests it's the valve rather than PPM/flow)."""
    if baseline_mg is None or followup_mg is None:
        return None, "no baseline+follow-up gradient pair available"
    delta_mg = followup_mg - baseline_mg

    def eoa_drop_ok(threshold_abs, threshold_pct):
        if baseline_eoa is None or followup_eoa is None or baseline_eoa <= 0:
            return False
        drop = baseline_eoa - followup_eoa
        return drop >= threshold_abs or (drop / baseline_eoa) >= threshold_pct

    def dvi_drop_ok(threshold_abs, threshold_pct):
        if baseline_dvi is None or followup_dvi is None or baseline_dvi <= 0:
            return False
        drop = baseline_dvi - followup_dvi
        return drop >= threshold_abs or (drop / baseline_dvi) >= threshold_pct

    if delta_mg >= VARC3_STAGE3_DELTA_MG and followup_mg >= VARC3_STAGE3_ABS_MG:
        if eoa_drop_ok(VARC3_STAGE3_EOA_DROP_CM2, VARC3_STAGE3_EOA_DROP_PCT) or \
           dvi_drop_ok(VARC3_STAGE3_DVI_DROP_ABS, VARC3_STAGE3_DVI_DROP_PCT):
            return 3, f"VARC-3 Stage 3: Δmean-gradient={delta_mg:.0f} mmHg (>=20), follow-up MG={followup_mg:.0f} (>=30), concurrent EOA/DVI drop confirmed"
        return None, f"gradient meets Stage-3 delta/absolute (Δ{delta_mg:.0f}, MG {followup_mg:.0f}) but no concurrent EOA/DVI drop — cannot rule out PPM/high-flow; downgrade to Capodanno fallback"

    if delta_mg >= VARC3_STAGE2_DELTA_MG and followup_mg >= VARC3_STAGE2_ABS_MG:
        if eoa_drop_ok(VARC3_STAGE2_EOA_DROP_CM2, VARC3_STAGE2_EOA_DROP_PCT) or \
           dvi_drop_ok(VARC3_STAGE2_DVI_DROP_ABS, VARC3_STAGE2_DVI_DROP_PCT):
            return 2, f"VARC-3 Stage 2: Δmean-gradient={delta_mg:.0f} mmHg (>=10), follow-up MG={followup_mg:.0f} (>=20), concurrent EOA/DVI drop confirmed"
        return None, f"gradient meets Stage-2 delta/absolute (Δ{delta_mg:.0f}, MG {followup_mg:.0f}) but no concurrent EOA/DVI drop — downgrade to Capodanno fallback"

    return 0, f"no VARC-3 delta threshold met (Δmean-gradient={delta_mg:.0f} mmHg)"


def classify_hvd_stage_capodanno(followup_mg, ar_ordinal=None):
    """Fallback used when no baseline echo is recoverable (the majority
    case here — architecture doc section 2, 'Λύση για την έλλειψη baseline
    echo'). Result is tagged 'probable', never 'definite', by the caller."""
    if followup_mg is None and ar_ordinal is None:
        return 0, "no post-implant hemodynamic data"
    reasons = []
    stage = 0
    if followup_mg is not None:
        if followup_mg >= CAPODANNO_SEVERE_MG:
            stage = max(stage, 3)
            reasons.append(f"MG={followup_mg:.0f} mmHg >= {CAPODANNO_SEVERE_MG} (severe, Capodanno absolute)")
        elif followup_mg >= CAPODANNO_MODERATE_MG:
            stage = max(stage, 2)
            reasons.append(f"MG={followup_mg:.0f} mmHg >= {CAPODANNO_MODERATE_MG} (moderate, Capodanno absolute)")
    if ar_ordinal is not None and ar_ordinal >= 2:
        stage = max(stage, 2 if ar_ordinal == 2 else 3)
        reasons.append(f"new moderate/severe intraprosthetic AR (ordinal={ar_ordinal})")
    if not reasons:
        reasons.append(f"MG={followup_mg} below moderate threshold and no significant AR")
    return stage, "; ".join(reasons)


def stage_phenotype_note(has_stenosis_evidence: bool, has_regurg_evidence: bool) -> str:
    if has_stenosis_evidence and has_regurg_evidence:
        return "RS"
    if has_regurg_evidence:
        return "R"
    if has_stenosis_evidence:
        return "S"
    return ""


CONFIDENCE_RANK = {"definite": 3, "probable": 2, "possible": 1, "excluded": 0, "censored": 0}


def assign_patient_label(evidence: dict) -> dict:
    """
    evidence: dict assembled by pipeline.py per patient, expected keys:
      has_exclusion (bool), exclusion_reason (str|None)
      has_redo (bool), redo_is_viv (bool), redo_note_year (int|None), redo_evidence (str)
      has_svd_explicit_text (bool), svd_explicit_evidence (str), svd_explicit_year (int|None)
      followup_gradients: list of {mean, peak, dvi, note_year} post-implant
      baseline_gradient: {mean, dvi} or None (pre-implant / early post-implant reference)
      max_ar_ordinal: int|None, ar_evidence: str|None
      has_morphology_only: bool (calcification/HALT/leaflet-motion mention, no hemodynamic change)
      morphology_evidence: str|None
      last_note_year: int
      index_implant_year: int|None

    Returns a dict: patient-level label record for labels.csv + reviewer sheet.
    """
    out = dict(
        hvd_stage=0, bvf_stage=0, phenotype="", confidence_tier="censored",
        rationale=[], right_censored_at_year=evidence.get("last_note_year"),
    )

    # --- Exclusion / competing-risk events take priority -------------------
    if evidence.get("has_exclusion"):
        out.update(confidence_tier="excluded",
                    rationale=[f"Competing event / exclusion: {evidence.get('exclusion_reason')}"])
        # A redo driven by endocarditis/thrombosis is still a BVF-3-adjacent
        # clinical event but NOT an SVD event; record it as excluded rather
        # than folding it into BVF Stage 2, per architecture doc section 3
        # ("Αποκλεισμός/ανταγωνιστικό συμβάν").
        return out

    rationale = []

    # --- Definite tier: reintervention (redo/ViV) with device-related cause
    if evidence.get("has_redo"):
        out["bvf_stage"] = 2
        out["confidence_tier"] = "definite"
        tag = "BVF Stage 2 (ViV re-intervention)" if evidence.get("redo_is_viv") else "BVF Stage 2 (surgical redo)"
        rationale.append(f"{tag}: {evidence.get('redo_evidence')}")

    # --- Definite tier: explicit SVD / prosthetic-failure language ---------
    if evidence.get("has_svd_explicit_text"):
        out["confidence_tier"] = "definite"
        rationale.append(f"Explicit SVD/prosthetic-failure text: {evidence.get('svd_explicit_evidence')}")

    # --- Hemodynamic staging -------------------------------------------------
    baseline = evidence.get("baseline_gradient")
    followups = evidence.get("followup_gradients") or []
    stage_from_hemo, hemo_conf, hemo_reason = 0, None, None
    worst_followup_mg = None
    if followups:
        worst = max(followups, key=lambda g: g.get("mean") or 0)
        worst_followup_mg = worst.get("mean")
        if baseline and baseline.get("mean") is not None:
            stage, reason = classify_hvd_stage_from_gradients(
                baseline["mean"], worst.get("mean"),
                baseline.get("eoa"), worst.get("eoa"),
                baseline.get("dvi"), worst.get("dvi"),
            )
            if stage is not None:
                # stage == 0 here means "no VARC-3 delta threshold met" — a
                # genuinely normal follow-up gradient, not evidence of
                # possible HVD/SVD. Only stage > 0 should ever raise the
                # confidence tier off "censored"; a normal reading is
                # right-censoring information, not a "possible" signal.
                stage_from_hemo, hemo_conf, hemo_reason = stage, ("definite" if stage > 0 else None), reason
            else:
                stage2, reason2 = classify_hvd_stage_capodanno(worst.get("mean"), evidence.get("max_ar_ordinal"))
                stage_from_hemo, hemo_conf = stage2, ("probable" if stage2 > 0 else None)
                hemo_reason = f"{reason} | Capodanno fallback: {reason2}"
        else:
            stage2, reason2 = classify_hvd_stage_capodanno(worst.get("mean"), evidence.get("max_ar_ordinal"))
            stage_from_hemo, hemo_conf = stage2, ("probable" if stage2 > 0 else None)
            hemo_reason = f"no baseline echo available — Capodanno absolute-threshold fallback: {reason2}"
        if stage_from_hemo > out["hvd_stage"]:
            out["hvd_stage"] = stage_from_hemo
        if hemo_reason:
            rationale.append(f"HVD staging: {hemo_reason}")
        if hemo_conf and CONFIDENCE_RANK.get(hemo_conf, 0) > CONFIDENCE_RANK.get(out["confidence_tier"], 0) \
                and out["confidence_tier"] not in ("definite",):
            out["confidence_tier"] = hemo_conf

    # If a redo/explicit-SVD definite trigger fired but we have no staged
    # hemodynamics to size it, still record it as HVD stage >=2 by
    # definition (an intervention implies at least Stage 2 hemodynamic
    # compromise triggered it) — VARC-3 links BVF Stage 2/3 to an underlying
    # HVD stage >=2/3.
    if out["confidence_tier"] == "definite" and out["hvd_stage"] < 2:
        out["hvd_stage"] = max(out["hvd_stage"], 2)
        rationale.append("HVD stage floor of 2 imputed from definite reintervention/explicit-SVD trigger (no separately staged gradient available)")

    # --- Possible tier: morphology-only, no hemodynamic change -------------
    if out["confidence_tier"] not in ("definite", "probable") and evidence.get("has_morphology_only"):
        out["confidence_tier"] = "possible"
        out["hvd_stage"] = max(out["hvd_stage"], 1)
        rationale.append(f"Possible/Stage-1 morphology-only finding: {evidence.get('morphology_evidence')}")

    # --- Possible tier: qualitative gradient worsening, no number given ----
    # ("increased AVR gradients" with no mmHg value) — can't be VARC-3
    # staged, but flagged for physician adjudication rather than censored.
    if out["confidence_tier"] not in ("definite", "probable", "possible") and evidence.get("has_gradient_trend_only"):
        out["confidence_tier"] = "possible"
        out["hvd_stage"] = max(out["hvd_stage"], 1)
        rationale.append(f"Possible/Stage-1 qualitative gradient worsening (no numeric value to stage): {evidence.get('gradient_trend_evidence')}")

    # --- Possible tier: SVD-language phrase found only as a bare coded
    # problem-list entry (e.g. "Prosthetic aortic valve failure" on its own
    # line in a Diagnosis/Date table), with no narrative elaboration and no
    # way to confirm SVD vs. PVL/endocarditis/thrombosis etiology from the
    # phrase alone (see review/error_catalogue.md, finding 3.2). Never
    # promoted to 'definite' — a human must adjudicate the true cause.
    if out["confidence_tier"] not in ("definite", "probable", "possible") and evidence.get("has_svd_history_list_only"):
        out["confidence_tier"] = "possible"
        out["hvd_stage"] = max(out["hvd_stage"], 1)
        rationale.append(f"Possible/coded-history-only SVD phrase (etiology unconfirmed, no narrative elaboration): {evidence.get('svd_history_list_evidence')}")

    # NOTE: has_stenosis_evidence must reflect an ACTUAL stenotic gradient
    # (worst_followup_mg at/above the Capodanno moderate threshold, or a
    # VARC-3-strict stenosis stage), not merely "a followup gradient value
    # exists" (bool(followups)) and not stage_from_hemo (which, via the
    # Capodanno-fallback path, is max(MG-driven stage, AR-driven stage) and
    # so can be >0 from AR alone with a normal MG). An earlier version used
    # bool(followups) and mislabeled several AR-only-driven patients (e.g.
    # worst_followup_mg=3-9 mmHg alongside a real AR finding) as phenotype
    # "RS" instead of "R" — caught by the Head B unit tests
    # (tests/test_varc3_rules.py), fixed here.
    has_stenosis_evidence = (
        (worst_followup_mg is not None and worst_followup_mg >= CAPODANNO_MODERATE_MG)
        or bool(evidence.get("has_morphology_only"))
        or bool(evidence.get("has_gradient_trend_only"))
    )
    out["phenotype"] = stage_phenotype_note(
        has_stenosis_evidence=has_stenosis_evidence,
        has_regurg_evidence=bool(evidence.get("max_ar_ordinal")),
    )
    out["rationale"] = rationale if rationale else ["No post-implant HVD/SVD evidence found; right-censored"]
    return out
