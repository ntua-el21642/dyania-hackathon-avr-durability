"""
Unit tests for the Head B VARC-3 rule engine (Master Prompt Section 5.5):
one synthetic case per stage (0, 1, 2S, 2R, 2RS, 3S, 3R) and per edge case
(PPM without concurrent EOA/DVI change, high-flow state, PVL-only,
endocarditis).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varc3_rules import assign_patient_label, classify_hvd_stage_from_gradients


def _base_evidence(**overrides):
    ev = dict(
        has_exclusion=False, exclusion_reason=None,
        has_redo=False, redo_is_viv=False, redo_note_year=None, redo_evidence=None,
        has_svd_explicit_text=False, svd_explicit_evidence=None, svd_explicit_year=None,
        has_svd_history_list_only=False, svd_history_list_evidence=None,
        followup_gradients=[], baseline_gradient=None,
        max_ar_ordinal=None, ar_evidence=None,
        has_morphology_only=False, morphology_evidence=None,
        has_gradient_trend_only=False, gradient_trend_evidence=None,
        last_note_year=2020, index_implant_year=2015,
    )
    ev.update(overrides)
    return ev


def test_stage_0_no_evidence_is_censored():
    label = assign_patient_label(_base_evidence())
    assert label["hvd_stage"] == 0
    assert label["bvf_stage"] == 0
    assert label["confidence_tier"] == "censored"


def test_stage_1_morphology_only_is_possible():
    label = assign_patient_label(_base_evidence(
        has_morphology_only=True, morphology_evidence="[Progress Notes 2019] leaflet thickening noted"))
    assert label["hvd_stage"] == 1
    assert label["confidence_tier"] == "possible"
    assert label["bvf_stage"] == 0


def test_stage_2s_stenosis_varc3_strict_with_eoa_drop():
    label = assign_patient_label(_base_evidence(
        baseline_gradient=dict(mean=10, dvi=0.45, note_year=2015),
        followup_gradients=[dict(mean=22, dvi=0.30, note_year=2019)],
    ))
    assert label["hvd_stage"] == 2
    assert label["phenotype"] == "S"
    assert "VARC-3 Stage 2" in " ".join(label["rationale"])


def test_stage_2r_regurgitation_capodanno_fallback_no_baseline():
    label = assign_patient_label(_base_evidence(
        max_ar_ordinal=2, ar_evidence="[Progress Notes 2019] moderate aortic regurgitation",
        followup_gradients=[dict(mean=8, dvi=0.5, note_year=2019)],
    ))
    assert label["hvd_stage"] == 2
    assert label["phenotype"] == "R"


def test_stage_2rs_combined_stenosis_and_regurgitation():
    label = assign_patient_label(_base_evidence(
        max_ar_ordinal=2, ar_evidence="[Progress Notes 2019] moderate AR",
        followup_gradients=[dict(mean=25, dvi=0.4, note_year=2019)],
    ))
    assert label["hvd_stage"] == 2
    assert label["phenotype"] == "RS"


def test_stage_3s_stenosis_varc3_strict_with_eoa_drop():
    label = assign_patient_label(_base_evidence(
        baseline_gradient=dict(mean=8, dvi=0.5, note_year=2015),
        followup_gradients=[dict(mean=32, dvi=0.25, note_year=2020)],
    ))
    assert label["hvd_stage"] == 3
    assert label["phenotype"] == "S"


def test_stage_3r_severe_regurgitation_capodanno_fallback():
    label = assign_patient_label(_base_evidence(
        max_ar_ordinal=3, ar_evidence="[Progress Notes 2019] severe AR",
        followup_gradients=[dict(mean=8, dvi=0.5, note_year=2019)],
    ))
    assert label["hvd_stage"] == 3
    assert label["phenotype"] == "R"


def test_bvf_stage_2_reintervention_is_definite():
    label = assign_patient_label(_base_evidence(
        has_redo=True, redo_is_viv=True, redo_note_year=2021,
        redo_evidence="[Progress Notes 2021] ViV TAVR performed",
    ))
    assert label["bvf_stage"] == 2
    assert label["confidence_tier"] == "definite"
    assert label["hvd_stage"] >= 2  # HVD stage floor imputed from the reintervention trigger


def test_ppm_without_concurrent_eoa_dvi_change_downgrades_to_capodanno():
    """A gradient rise meeting the VARC-3 Stage-2 delta/absolute threshold
    but with NO concurrent EOA/DVI drop cannot rule out patient-prosthesis
    mismatch or a high-flow state -- must downgrade to the Capodanno
    fallback (probable), not assert a VARC-3-strict (definite-confidence
    hemodynamic) stage."""
    stage, reason = classify_hvd_stage_from_gradients(
        baseline_mg=8, followup_mg=22, baseline_eoa=1.2, followup_eoa=1.15,  # EOA barely moved
        baseline_dvi=0.45, followup_dvi=0.43,  # DVI barely moved
    )
    assert stage is None
    assert "no concurrent EOA/DVI drop" in reason


def test_high_flow_state_same_downgrade_path_as_ppm():
    """High-output/high-flow states can also raise transvalvular gradients
    without a concurrent EOA/DVI drop -- same VARC-3 safeguard, tested
    separately from the PPM case above because the clinical mechanism
    differs even though the rule-engine code path is identical (this is
    intentional: VARC-3 cannot distinguish PPM from high-flow from the
    gradient/EOA/DVI numbers alone, and correctly refuses to assert
    'definite' SVD in either case)."""
    stage, reason = classify_hvd_stage_from_gradients(
        baseline_mg=15, followup_mg=35, baseline_eoa=1.0, followup_eoa=0.98,
        baseline_dvi=0.4, followup_dvi=0.39,
    )
    assert stage is None
    assert "no concurrent EOA/DVI drop" in reason


def test_pvl_only_is_excluded_not_svd():
    label = assign_patient_label(_base_evidence(
        has_exclusion=True, exclusion_reason="redo/ViV in 2020 attributed to: pvl ('paravalvular leak repair')",
    ))
    assert label["confidence_tier"] == "excluded"
    assert label["bvf_stage"] == 0
    assert label["hvd_stage"] == 0


def test_endocarditis_is_excluded_not_svd():
    label = assign_patient_label(_base_evidence(
        has_exclusion=True, exclusion_reason="redo/ViV in 2020 attributed to: endocarditis ('active endocarditis')",
        has_redo=True, redo_is_viv=False, redo_note_year=2020,  # a redo DID occur, but must not count as SVD/BVF
    ))
    assert label["confidence_tier"] == "excluded"
    assert label["bvf_stage"] == 0


def test_svd_history_list_only_is_possible_never_definite():
    """Regression test for review/error_catalogue.md finding 3.2: a bare
    coded-history-list SVD phrase must never reach 'definite' on its own."""
    label = assign_patient_label(_base_evidence(
        has_svd_history_list_only=True,
        svd_history_list_evidence="[Progress Notes 2022] Prosthetic aortic valve failure",
    ))
    assert label["confidence_tier"] == "possible"
    assert label["confidence_tier"] != "definite"
