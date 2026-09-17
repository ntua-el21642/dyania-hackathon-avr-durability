"""Single-page valve durability review."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st
import yaml

from ui.charts import valve_outlook_curve
from ui.data import AuditSummary, ClinicalDataRepository, PatientRecord
from ui.predictor import DurabilityPrediction, load_posterior, predict_durability
from ui.reporting import build_research_summary
from ui.styles import APP_CSS, safe, section_header


ROOT = Path(__file__).resolve().parent
POSTERIOR_PATH = ROOT / "reports" / "head_a_posterior.nc"
SAMPLE_NOTES_PATH = ROOT / "samples" / "patient_030_notes.txt"
EVIDENCE_COLUMNS = (
    ("Reintervention evidence", "redo_evidence"),
    ("Explicit SVD evidence", "svd_explicit_evidence"),
    ("Regurgitation evidence", "ar_evidence"),
    ("Morphology evidence", "morphology_evidence"),
    ("Gradient-trend evidence", "gradient_trend_evidence"),
    ("Alternative mechanism evidence", "exclusion_reason"),
)

st.set_page_config(page_title="ValveVie | Valve Durability", page_icon="V", layout="wide", initial_sidebar_state="collapsed")
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_repository(root: str) -> ClinicalDataRepository:
    return ClinicalDataRepository(root)


@st.cache_resource(show_spinner="Loading durability model…")
def get_posterior(path: str):
    return load_posterior(path)


def parse_text_notes(text: str) -> pd.DataFrame:
    """Parse ValveVie's small, readable longitudinal-note text format."""

    text = text.lstrip("\ufeff")
    records = []
    for block in re.split(r"\r?\n\s*--- NOTE ---\s*\r?\n", text.strip()):
        match = re.match(
            r"PROFILE KEY:\s*(.+?)\s*\r?\nTYPE:\s*(.+?)\s*\r?\nSERVICE DATE:\s*(\d{4})\s*\r?\nNOTES:\s*\r?\n([\s\S]+)",
            block.strip(),
            re.IGNORECASE,
        )
        if not match:
            raise ValueError("Text notes must contain Profile Key, Type, Service Date and Notes fields.")
        records.append({"Profile Key": match.group(1).strip(), "Type": match.group(2).strip(),
                        "Service Date": int(match.group(3)), "Notes": match.group(4).strip()})
    return pd.DataFrame(records)


@st.cache_data(show_spinner="Reading notes…")
def process_notes_file(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read an uploaded notes file and run the deterministic NLP pipeline."""

    from pipeline import run_pipeline

    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        from io import BytesIO
        frame = pd.read_excel(BytesIO(file_bytes))
    elif suffix == ".csv":
        from io import BytesIO
        frame = pd.read_csv(BytesIO(file_bytes))
    elif suffix == ".txt":
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = file_bytes.decode("cp1252")
        frame = parse_text_notes(text)
    else:
        raise ValueError("Supported formats are XLSX, CSV and TXT.")

    required = {"Profile Key", "Type", "Notes", "Service Date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {', '.join(sorted(required - set(frame.columns)))}")

    return run_pipeline(frame)


def optional_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def family_mapping() -> dict[str, str]:
    path = ROOT / "config" / "priors.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return (yaml.safe_load(stream) or {}).get("valve_model_to_family", {})


def patient_from_pipeline(row: pd.Series) -> PatientRecord:
    valve_model = optional_text(row.get("index_valve_model"))
    valve_family = family_mapping().get(valve_model) if valve_model else None
    size = optional_float(row.get("index_valve_size_mm"))
    index_year = optional_float(row.get("index_implant_year"))
    last_year = optional_float(row.get("last_note_year"))
    bvf_stage = int(row.get("bvf_stage", 0) or 0)
    evidence = tuple(
        (label, text)
        for label, column in EVIDENCE_COLUMNS
        if (text := optional_text(row.get(column))) is not None
    )
    return PatientRecord(
        profile_key=str(row["profile_key"]),
        index_implant_year=int(index_year) if index_year is not None else None,
        index_implant_source=optional_text(row.get("index_implant_source")),
        approach=optional_text(row.get("index_approach")),
        valve_model=valve_model,
        valve_family=valve_family,
        valve_family_known=valve_family is not None,
        valve_size_mm=size,
        size_known=size is not None,
        ppm_proxy_flag=size is not None and size <= 21,
        last_note_year=int(last_year) if last_year is not None else None,
        years_followup=optional_float(row.get("years_followup")),
        hvd_stage=int(row.get("hvd_stage", 0) or 0),
        bvf_stage=bvf_stage,
        phenotype=optional_text(row.get("phenotype")),
        confidence_tier=optional_text(row.get("confidence_tier")) or "unknown",
        rationale=optional_text(row.get("rationale")) or "No rationale available.",
        event=bvf_stage >= 2,
        t_lower=optional_float(row.get("years_followup")),
        t_upper=None,
        baseline_mg=optional_float(row.get("baseline_mg")),
        worst_followup_mg=optional_float(row.get("worst_followup_mg")),
        evidence=evidence,
    )


def audit_from_pipeline(frame: pd.DataFrame, profile_key: str) -> AuditSummary:
    rows = frame[frame["profile_key"].astype(str) == profile_key]

    def total(column: str) -> int:
        return int(pd.to_numeric(rows.get(column, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    post = rows.get("is_post_implant", pd.Series(False, index=rows.index)).astype(bool)
    return AuditSummary(
        total_notes=len(rows), post_implant_notes=int(post.sum()), redo_hits=total("n_redo"),
        viv_hits=total("n_viv"), svd_hits=total("n_svd_explicit"), gradient_hits=total("n_gradients"),
        morphology_hits=total("n_morphology"), exclusion_hits=total("n_exclusion"),
        note_types=tuple(sorted(rows.get("note_type", pd.Series(dtype=str)).dropna().astype(str).unique())),
    )


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def risk_category(risk: float, patient: PatientRecord) -> str:
    if risk > 0.15 or patient.hvd_stage >= 3:
        return "High"
    if risk >= 0.05 or patient.hvd_stage == 2:
        return "Moderate"
    return "Low"


def follow_up_plan(risk: float, patient: PatientRecord) -> tuple[str, str, str, float | None]:
    flags = tuple(flag for flag in patient.alternative_mechanism_flags() if flag.detected)
    thrombosis = any(flag.label == "Thrombosis / HALT" for flag in flags)
    rapid_gradient = patient.gradient_delta is not None and patient.gradient_delta >= 10
    if patient.event or patient.bvf_stage >= 2:
        return "Review now", "Confirm recorded endpoint", "Valve-failure or reintervention event found", None
    if risk > 0.15 or patient.hvd_stage >= 3 or rapid_gradient:
        test = "TTE and specialist valve review"
        if thrombosis:
            test += "; consider 4D-CT/TEE for suspected thrombosis"
        return "Within 6 months", test, "High-risk or rapid-change pathway", 0.5
    if risk >= 0.05 or patient.hvd_stage == 2:
        return "Within 12 months", "Transthoracic echocardiogram (TTE)", "Moderate risk or Stage 2 HVD", 1.0
    return "Standard schedule", "TTE per treating team's surveillance plan", "No model-driven shortening", 2.0


def stage_track(patient: PatientRecord) -> str:
    names = ("No detected deterioration", "Morphological change", "Moderate HVD", "Severe HVD")
    return '<div class="stage-progress">' + "".join(
        f'<div class="stage-step {"active" if index == patient.hvd_stage else ""}"><div class="stage-dot">{index}</div><div class="stage-name">{safe(name)}</div></div>'
        for index, name in enumerate(names)
    ) + "</div>"


def patient_facts(patient: PatientRecord) -> str:
    size = f"{patient.valve_size_mm:g} mm" if patient.valve_size_mm is not None else "Unavailable"
    facts = (("Approach", patient.approach or "Unavailable"), ("Valve", patient.valve_model or "Unavailable"),
             ("Size", size), ("Implant year", patient.index_implant_year or "Unavailable"),
             ("Latest note", patient.last_note_year or "Unavailable"), ("Evidence", patient.confidence_tier.title()))
    return '<div class="facts">' + "".join(
        f'<div class="fact">{safe(label)}<strong>{safe(value)}</strong></div>'
        for label, value in facts
    ) + "</div>"


def factor_rows(prediction: DurabilityPrediction) -> str:
    rows = []
    for factor in prediction.factors:
        ratio = factor.ratio
        if factor.key == "approach" and abs(ratio.median - 1.0) < 0.01:
            direction = "Reference category"
        elif not factor.applied:
            direction = "Reference value"
        elif ratio.median < 0.98:
            direction = "Reduces modeled durability"
        elif ratio.median > 1.02:
            direction = "Increases modeled durability"
        else:
            direction = "Minimal change"
        description = {
            "approach": "Compared with the SAVR reference.",
            "family": factor.description if factor.applied else "Family not represented; approach estimate used.",
            "ppm": "Small-valve adjustment applied." if factor.applied else "No small-valve adjustment.",
        }.get(factor.key, factor.description)
        rows.append(
            f'<div class="factor"><div class="factor-name">{safe(factor.label)}</div><div class="factor-copy">{safe(description)}</div>'
            f'<div class="factor-effect">{safe(direction)}<small>{ratio.median:.2f}× · range {ratio.low:.2f}–{ratio.high:.2f}</small></div></div>'
        )
    return "".join(rows)


def detected_flag_rows(patient: PatientRecord) -> str:
    flags = tuple(flag for flag in patient.alternative_mechanism_flags() if flag.detected)
    return "".join(
        f'<div class="flag"><div class="flag-dot"></div><div><div class="flag-name">{safe(flag.label)}</div>'
        f'<div class="flag-copy">{safe(flag.source or flag.next_step)}</div></div><div class="flag-status">Review</div></div>'
        for flag in flags
    )


def simple_stage_reason(patient: PatientRecord) -> str:
    parts = []
    if patient.worst_followup_mg is not None:
        parts.append(f"Mean gradient {patient.worst_followup_mg:g} mmHg")
    if any(label == "Explicit SVD evidence" for label, _ in patient.evidence):
        year = f" in {patient.last_note_year}" if patient.last_note_year else ""
        parts.append(f"prosthetic valve dysfunction documented{year}")
    if not parts:
        return patient.current_state_label
    return ". ".join(parts) + "."


repository = get_repository(str(ROOT))
results_mode = bool(st.session_state.get("results_mode"))
if not results_mode:
    st.markdown('<div class="nav"><div class="brand"><span class="brand-mark"></span>ValveVie</div><div class="nav-note">Clinical decision support</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="landing"><div class="landing-kicker">Valve intelligence</div><div class="landing-title">See what the record says about valve durability.</div><div class="landing-copy">Upload de-identified longitudinal notes to begin.</div></div>', unsafe_allow_html=True)

    with st.form("notes_upload", border=False):
        uploaded = st.file_uploader("Patient notes", type=["xlsx", "csv", "txt"], label_visibility="collapsed")
        submitted = st.form_submit_button("View results", type="primary", use_container_width=True)

    download_col, format_col = st.columns([1, 2.2], vertical_alignment="center")
    with download_col:
        st.download_button("Download sample notes", data=SAMPLE_NOTES_PATH.read_bytes(), file_name=SAMPLE_NOTES_PATH.name, mime="text/plain", use_container_width=True)
    with format_col:
        st.caption("Accepted formats: TXT, CSV, XLSX")

    if submitted and uploaded is None:
        st.warning("Choose a notes file first.")
    if submitted and uploaded is not None:
        st.session_state.notes_bytes = uploaded.getvalue()
        st.session_state.notes_name = uploaded.name
        st.session_state.results_mode = True
        st.rerun()
    st.markdown('<div class="landing-foot">De-identified data only</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="nav"><div class="brand"><span class="brand-mark"></span>ValveVie</div><div class="nav-links"><a class="active" href="#assessment">Assessment</a><a href="#outlook">Outlook</a><a href="#follow-up">Follow-up</a></div></div>', unsafe_allow_html=True)

try:
    uploaded_labels, uploaded_audit = process_notes_file(st.session_state.notes_bytes, st.session_state.notes_name)
    eligible = uploaded_labels[uploaded_labels["index_approach"].notna()]
    if eligible.empty:
        st.error("No dated AVR implant was found in these notes.")
        st.stop()
    row = eligible.iloc[0]
    patient = patient_from_pipeline(row)
    audit = audit_from_pipeline(uploaded_audit, patient.profile_key)
except Exception as exc:
    if isinstance(exc, UnicodeError):
        message = "The text encoding is not supported. Save the file as UTF-8 and try again."
    elif isinstance(exc, ValueError):
        message = str(exc)
    elif "openpyxl" in str(exc).lower():
        message = "Excel support is not installed. Run: pip install -r requirements-ui.txt"
    else:
        message = "The notes could not be processed."
    st.error(message)
    with st.expander("Technical detail"):
        st.code(str(exc))
    st.stop()

st.markdown(
    f'<div class="case-head" id="assessment"><div><div class="case-title">Valve durability assessment</div>'
    f'<div class="case-meta">{safe(patient.profile_key)} · {safe(patient.approach or "Approach unavailable")} · {safe(patient.valve_model or "Valve unavailable")} · {safe(patient.years_followup or 0)} years observed</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

prediction: DurabilityPrediction | None = None
prediction_error: str | None = None
if not patient.event and patient.bvf_stage < 2:
    try:
        prediction = predict_durability(get_posterior(str(POSTERIOR_PATH)), approach=patient.approach or "", valve_family=patient.valve_family,
                                        family_known=patient.valve_family_known, ppm_proxy_flag=patient.ppm_proxy_flag,
                                        size_known=patient.size_known, observed_event_free_years=patient.years_followup or 0.0)
    except Exception as exc:
        prediction_error = str(exc)

if prediction is None:
    st.warning("A future durability estimate is unavailable because an endpoint is recorded or the extracted implant inputs are incomplete.")
    if prediction_error:
        with st.expander("Technical detail"):
            st.code(prediction_error)
else:
    five_year = prediction.conditional_risk_by_horizon[5]
    category = risk_category(five_year.median, patient)
    interval, examination, basis, _review_year = follow_up_plan(five_year.median, patient)
    remaining = prediction.remaining_median_years
    st.markdown(
        '<div class="summary">'
        f'<div class="summary-main"><div class="eyebrow">Estimated remaining durability</div><div class="hero-value">{remaining.median:.1f} years</div><div class="detail">From latest follow-up · likely range {remaining.low:.1f}–{remaining.high:.1f} years</div></div>'
        f'<div class="summary-side"><div class="eyebrow">Current valve state</div><div class="side-value">Stage {patient.hvd_stage}</div><div class="detail">{safe(patient.current_state_label)}</div></div>'
        f'<div class="summary-side"><div class="eyebrow">Risk within 5 years</div><div class="side-value">{pct(five_year.median)}</div><div class="detail">Estimated chance of reaching the study\'s valve-failure endpoint.</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="content-section outlook-first" id="outlook">' + section_header("Durability outlook", "Higher on the chart means a greater chance of remaining free from valve failure"), unsafe_allow_html=True)
    st.plotly_chart(valve_outlook_curve(prediction, median_year=remaining.median), width="stretch", config={"displayModeBar": False, "responsive": True})
    st.markdown('<div class="chart-guide"><span><i class="legend-line"></i>Patient</span><span><i class="legend-line grey"></i>Reference</span><span><i class="legend-band"></i>Likely range</span></div><div class="method-note">The vertical line marks the median model estimate. It does not predict or recommend surgery.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section">' + section_header("Drivers of the estimate", "What moved the prediction"), unsafe_allow_html=True)
    st.markdown('<div class="factor-list">' + factor_rows(prediction) + '</div><div class="method-note">Below 1× suggests shorter durability; above 1× suggests longer durability.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section">' + section_header("Valve status", "Current hemodynamic deterioration stage"), unsafe_allow_html=True)
    st.markdown(stage_track(patient), unsafe_allow_html=True)
    st.markdown(patient_facts(patient), unsafe_allow_html=True)
    st.markdown(f'<div class="evidence"><strong>Why:</strong> {safe(simple_stage_reason(patient))}</div></div>', unsafe_allow_html=True)

    detected_flags = detected_flag_rows(patient)
    if detected_flags:
        st.markdown('<div class="content-section">' + section_header("Additional findings", "Alternative mechanisms detected in the notes"), unsafe_allow_html=True)
        st.markdown(detected_flags + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-section" id="follow-up">' + section_header("Next review", "Treating clinician confirms or overrides"), unsafe_allow_html=True)
    st.markdown(
        '<div class="followup">'
        f'<div class="followup-item"><div class="followup-label">When</div><div class="followup-value">{safe(interval)}</div></div>'
        f'<div class="followup-item"><div class="followup-label">Assessment</div><div class="followup-value">{safe(examination)}</div></div>'
        f'<div class="followup-item"><div class="followup-label">Clinical basis</div><div class="followup-value">{safe(basis)}</div></div></div>'
        '<div class="followup-foot">Clinician confirmation required.</div></div>',
        unsafe_allow_html=True,
    )

with st.expander("How reliable is this estimate?"):
    cohort = repository.cohort_summary()
    st.markdown(
        '<div class="evidence-strip">'
        f'<div><strong>{audit.total_notes}</strong><span>notes reviewed</span></div>'
        f'<div><strong>{cohort["events"]}</strong><span>valve-failure events informed the model</span></div>'
        '<div><strong>Wide</strong><span>uncertainty around individual estimates</span></div></div>'
        '<div class="method-note">This is an early research model. Physician adjudication is still in progress and the estimate should support—not replace—clinical review.</div>',
        unsafe_allow_html=True,
    )
summary = build_research_summary(patient, prediction, audit)
st.download_button("Download case summary", data=summary, file_name=f"{patient.profile_key.lower()}_valve_review.md", mime="text/markdown")
st.markdown('<div class="footer-note">ValveVie · Valve durability decision support</div>', unsafe_allow_html=True)
