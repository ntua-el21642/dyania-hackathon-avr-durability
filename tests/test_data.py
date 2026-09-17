from pathlib import Path

import pandas as pd
import yaml

from ui.data import ClinicalDataRepository


def write_fixture(root: Path) -> None:
    pd.DataFrame(
        [
            {
                "profile_key": "Patient_001",
                "index_implant_year": 2022,
                "index_implant_source": "op_report_service_date",
                "index_valve_model": "Perimount",
                "index_valve_size_mm": 25,
                "index_approach": "SAVR",
                "last_note_year": 2025,
                "years_followup": 3,
                "hvd_stage": 2,
                "bvf_stage": 0,
                "phenotype": "S",
                "confidence_tier": "probable",
                "rationale": "Capodanno absolute-threshold fallback",
                "redo_evidence": None,
                "svd_explicit_evidence": None,
                "exclusion_reason": None,
                "ar_evidence": None,
                "morphology_evidence": None,
                "gradient_trend_evidence": None,
                "worst_followup_mg": 38,
                "baseline_mg": None,
            }
        ]
    ).to_csv(root / "labels.csv", index=False)

    pd.DataFrame(
        [
            {
                "profile_key": "Patient_001",
                "index_implant_year": 2022,
                "index_implant_source": "op_report_service_date",
                "index_approach": "SAVR",
                "index_valve_model": "Perimount",
                "valve_family": "known_family",
                "valve_family_known": True,
                "index_valve_size_mm": 25,
                "size_known": True,
                "ppm_proxy_flag": False,
            }
        ]
    ).to_csv(root / "data_processed_patient_features.csv", index=False)

    pd.DataFrame(
        [
            {
                "profile_key": "Patient_001",
                "event": 0,
                "t_lower": 3,
                "t_upper": float("inf"),
                "confidence_tier": "probable",
                "hvd_stage": 2,
                "bvf_stage": 0,
            }
        ]
    ).to_csv(root / "data_processed_patient_labels.csv", index=False)

    pd.DataFrame(
        [
            {
                "profile_key": "Patient_001",
                "note_type": "Operative Report",
                "service_date": 2022,
                "is_post_implant": False,
                "n_redo": 0,
                "n_viv": 0,
                "n_svd_explicit": 0,
                "n_gradients": 0,
                "n_exclusion": 0,
                "n_morphology": 0,
            },
            {
                "profile_key": "Patient_001",
                "note_type": "Progress Notes",
                "service_date": 2025,
                "is_post_implant": True,
                "n_redo": 0,
                "n_viv": 0,
                "n_svd_explicit": 0,
                "n_gradients": 1,
                "n_exclusion": 0,
                "n_morphology": 0,
            },
        ]
    ).to_csv(root / "notes_audit.csv", index=False)

    reports = root / "reports"
    reports.mkdir()
    (reports / "head_a_bootstrap_bayesian.yaml").write_text(
        yaml.safe_dump(
            {
                "apparent_c_index": 0.611,
                "bootstrap_corrected_c_index": 0.588,
                "ci_95_low": 0.417,
                "ci_95_high": 0.735,
                "n_resamples_used": 200,
                "method_note": "Fixture method note",
            }
        ),
        encoding="utf-8",
    )


def test_repository_assembles_patient_and_audit(tmp_path):
    write_fixture(tmp_path)
    repository = ClinicalDataRepository(tmp_path)

    patient = repository.patient("Patient_001")
    audit = repository.audit_summary("Patient_001")
    validation = repository.validation_summary()

    assert patient.approach == "SAVR"
    assert patient.valve_family == "known_family"
    assert patient.endpoint_label == "HVD Stage 2"
    assert patient.phenotype_label == "Stenosis-dominant"
    assert audit.total_notes == 2
    assert audit.post_implant_notes == 1
    assert audit.gradient_hits == 1
    assert validation.corrected_c_index == 0.588

