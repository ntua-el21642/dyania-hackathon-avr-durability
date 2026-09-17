import numpy as np

from ui.data import AuditSummary, PatientRecord
from ui.predictor import PosteriorDraws, predict_durability
from ui.reporting import build_research_summary


def test_export_contains_scientific_boundaries():
    patient = PatientRecord(
        profile_key="Patient_001",
        index_implant_year=2022,
        index_implant_source="op_report_service_date",
        approach="SAVR",
        valve_model="Perimount",
        valve_family="known_family",
        valve_family_known=True,
        valve_size_mm=25.0,
        size_known=True,
        ppm_proxy_flag=False,
        last_note_year=2025,
        years_followup=3.0,
        hvd_stage=2,
        bvf_stage=0,
        phenotype="S",
        confidence_tier="probable",
        rationale="Example rationale",
        event=False,
        t_lower=3.0,
        t_upper=float("inf"),
        baseline_mg=None,
        worst_followup_mg=38.0,
        evidence=(),
    )
    posterior = PosteriorDraws(
        shape_k=np.full(50, 2.0),
        approach_effects={"SAVR": np.full(50, np.log(10.0))},
        family_offsets={"known_family": np.zeros(50)},
        beta_ppm=np.zeros(50),
    )
    prediction = predict_durability(
        posterior,
        approach="SAVR",
        valve_family="known_family",
        family_known=True,
        ppm_proxy_flag=False,
        size_known=True,
    )
    audit = AuditSummary(2, 1, 0, 0, 0, 1, 0, 0, ("Operative Report", "Progress Notes"))

    report = build_research_summary(patient, prediction, audit)

    assert "Not validated for clinical use" in report
    assert "Competing all-cause mortality is not modeled" in report
    assert "5-year modeled endpoint probability" in report
