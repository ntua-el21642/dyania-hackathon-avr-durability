"""Proof-of-concept research-summary export."""
from __future__ import annotations

from datetime import datetime, timezone

from .data import AuditSummary, PatientRecord
from .predictor import DurabilityPrediction


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _interval_pct(median: float, low: float, high: float) -> str:
    return f"{_pct(median)} (89% CrI {_pct(low)}–{_pct(high)})"


def build_research_summary(
    patient: PatientRecord,
    prediction: DurabilityPrediction | None,
    audit: AuditSummary,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    size = f"{patient.valve_size_mm:g} mm" if patient.valve_size_mm is not None else "Unavailable"
    lines = [
        "# ValveVie Case Summary",
        "",
        f"Generated: {generated}",
        "",
        "> Research prototype. Not validated for clinical use or patient-management decisions.",
        "",
        "## De-identified case",
        "",
        f"- Profile: {patient.profile_key}",
        f"- Index implant: {patient.index_implant_year or 'Unavailable'}",
        f"- Approach: {patient.approach or 'Unavailable'}",
        f"- Valve: {patient.valve_model or 'Unavailable'} ({size})",
        f"- PPM proxy: {'Yes' if patient.ppm_proxy_flag else 'No / unavailable'}",
        "",
        "## Current valve state (Head B)",
        "",
        f"- HVD stage: {patient.hvd_stage}",
        f"- BVF stage: {patient.bvf_stage}",
        f"- Phenotype: {patient.phenotype_label}",
        f"- Confidence tier: {patient.confidence_tier}",
        f"- Rationale: {patient.rationale}",
    ]

    if prediction is not None:
        risk_5 = prediction.conditional_risk_by_horizon.get(5)
        lines.extend(["", "## Durability forecast (Head A)", ""])
        if risk_5:
            lines.append(
                "- Modeled endpoint probability over the next 5 years: "
                + _interval_pct(risk_5.median, risk_5.low, risk_5.high)
            )
        m = prediction.median_event_free_years
        remaining = prediction.remaining_median_years
        lines.append(
            f"- Median remaining event-free time from latest follow-up: {remaining.median:.1f} years "
            f"(89% CrI {remaining.low:.1f}–{remaining.high:.1f})"
        )
        lines.append(
            f"- Posterior median event-free time: {m.median:.1f} years "
            f"(89% CrI {m.low:.1f}–{m.high:.1f})"
        )
        lines.append(f"- Posterior draws used: {prediction.posterior_draws:,}")
        lines.extend(["", "### Patient-specific AFT decomposition", ""])
        for factor in prediction.factors:
            r = factor.ratio
            lines.append(
                f"- {factor.label}: {r.median:.2f}x "
                f"(89% CrI {r.low:.2f}–{r.high:.2f}x). {factor.description}"
            )
    else:
        lines.extend(
            [
                "",
                "## Durability forecast (Head A)",
                "",
                "Posterior prediction unavailable for this profile.",
            ]
        )

    lines.extend(
        [
            "",
            "## Traceability",
            "",
            f"- Notes reviewed by pipeline: {audit.total_notes}",
            f"- Post-implant notes: {audit.post_implant_notes}",
            f"- Numeric gradient hits: {audit.gradient_hits}",
        ]
    )
    for label, text in patient.evidence:
        lines.append(f"- {label}: {text}")

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Head A is a hierarchical Bayesian Weibull AFT proof of concept trained on a very small event set.",
            "- Estimates are associations, not causal effects.",
            "- Competing all-cause mortality is not modeled in the current Head A; curves are not cumulative-incidence functions.",
            "- Full-cohort blind physician adjudication is pending.",
            "- This summary does not recommend a surveillance interval or treatment.",
            "",
            "Grounded in the repository's VARC-3/Capodanno phenotyping pipeline and Head A posterior artifact.",
        ]
    )
    return "\n".join(lines) + "\n"
