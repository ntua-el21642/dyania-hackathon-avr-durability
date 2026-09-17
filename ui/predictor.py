"""Posterior prediction utilities for the primary Bayesian Weibull AFT model.

This module deliberately contains no Streamlit code.  It is the narrow contract
between the fitted Head A artifact and the presentation layer, so the dashboard
does not depend on PyMC model internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


CI_LOW = 0.055
CI_HIGH = 0.945


class PredictionUnavailable(RuntimeError):
    """Raised when a patient cannot be scored by the fitted posterior."""


@dataclass(frozen=True)
class PosteriorDraws:
    """Flattened posterior draws required for patient-level inference."""

    shape_k: np.ndarray
    approach_effects: Mapping[str, np.ndarray]
    family_offsets: Mapping[str, np.ndarray]
    beta_ppm: np.ndarray

    @property
    def n_draws(self) -> int:
        return int(self.shape_k.size)


@dataclass(frozen=True)
class IntervalSummary:
    median: float
    low: float
    high: float


@dataclass(frozen=True)
class TimeRatioFactor:
    key: str
    label: str
    description: str
    ratio: IntervalSummary
    applied: bool = True


@dataclass(frozen=True)
class DurabilityPrediction:
    times: np.ndarray
    survival_median: np.ndarray
    survival_low: np.ndarray
    survival_high: np.ndarray
    risk_median: np.ndarray
    risk_low: np.ndarray
    risk_high: np.ndarray
    forecast_times: np.ndarray
    conditional_event_free_median: np.ndarray
    conditional_event_free_low: np.ndarray
    conditional_event_free_high: np.ndarray
    reference_event_free_median: np.ndarray
    conditional_risk_by_horizon: Mapping[int, IntervalSummary]
    landmark_years: float
    remaining_median_years: IntervalSummary
    median_event_free_years: IntervalSummary
    scale_years: IntervalSummary
    risk_by_year: Mapping[int, IntervalSummary]
    factors: tuple[TimeRatioFactor, ...]
    total_time_ratio: IntervalSummary
    posterior_draws: int
    family_effect_applied: bool
    ppm_effect_applied: bool


def _flatten(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _same_length(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    n = min(a.size for a in arrays)
    if n == 0:
        raise PredictionUnavailable("The posterior artifact contains no usable draws.")
    return tuple(a[:n] for a in arrays)


def _summary(values: np.ndarray) -> IntervalSummary:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        raise PredictionUnavailable("The posterior produced no finite values.")
    median, low, high = np.nanquantile(values, [0.5, CI_LOW, CI_HIGH])
    return IntervalSummary(float(median), float(low), float(high))


def load_posterior(path: str | Path) -> PosteriorDraws:
    """Load the small subset of an ArviZ NetCDF artifact needed by the UI.

    ArviZ is imported lazily so unit tests and Head B-only use do not require
    the Bayesian modeling stack.
    """

    posterior_path = Path(path)
    if not posterior_path.exists():
        raise PredictionUnavailable(f"Posterior artifact not found: {posterior_path}")

    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PredictionUnavailable(
            "ArviZ is required to read reports/head_a_posterior.nc. "
            "Install requirements-ui.txt."
        ) from exc

    try:
        posterior = az.from_netcdf(posterior_path).posterior
        shape_k = _flatten(posterior["shape_k"].values)
        beta_ppm = _flatten(posterior["beta_ppm"].values)

        approach_da = posterior["mu_approach"]
        approach_effects = {
            str(name): _flatten(approach_da.sel(approach=name).values)
            for name in approach_da.coords["approach"].values.tolist()
        }

        family_offsets: dict[str, np.ndarray] = {}
        if "family_offset" in posterior:
            family_da = posterior["family_offset"]
            family_offsets = {
                str(name): _flatten(family_da.sel(family=name).values)
                for name in family_da.coords["family"].values.tolist()
            }
    except Exception as exc:  # pragma: no cover - corrupted artifact path
        raise PredictionUnavailable(
            f"Could not read the Head A posterior artifact: {exc}"
        ) from exc

    shape_k, beta_ppm = _same_length(shape_k, beta_ppm)
    return PosteriorDraws(
        shape_k=shape_k,
        approach_effects=approach_effects,
        family_offsets=family_offsets,
        beta_ppm=beta_ppm,
    )


def _factor_summary(draws: np.ndarray) -> IntervalSummary:
    return _summary(np.exp(draws))


def predict_durability(
    posterior: PosteriorDraws,
    *,
    approach: str,
    valve_family: str | None,
    family_known: bool,
    ppm_proxy_flag: bool,
    size_known: bool,
    horizon_years: int = 15,
    observed_event_free_years: float = 0.0,
    forecast_horizon_years: int = 10,
    grid_points: int = 181,
) -> DurabilityPrediction:
    """Generate a patient-level posterior durability forecast.

    The current fitted model treats an unknown valve family as a zero family
    offset and an unknown labelled size as PPM-proxy off.  The return value
    exposes both decisions so the UI can state them rather than hiding them.
    """

    if not approach or approach not in posterior.approach_effects:
        available = ", ".join(sorted(posterior.approach_effects))
        raise PredictionUnavailable(
            f"Approach {approach!r} is not represented in the fitted model "
            f"(available: {available})."
        )

    k = _flatten(posterior.shape_k)
    mu = _flatten(posterior.approach_effects[approach])
    beta = _flatten(posterior.beta_ppm)
    k, mu, beta = _same_length(k, mu, beta)
    n = k.size

    family_applied = bool(
        family_known and valve_family and valve_family in posterior.family_offsets
    )
    family_offset = (
        _flatten(posterior.family_offsets[str(valve_family)])[:n]
        if family_applied
        else np.zeros(n, dtype=float)
    )
    if family_offset.size < n:
        k, mu, beta, family_offset = _same_length(k, mu, beta, family_offset)
        n = k.size

    ppm_applied = bool(size_known and ppm_proxy_flag)
    ppm_term = beta[:n] * float(ppm_applied)
    log_scale = mu[:n] + family_offset[:n] + ppm_term
    scale = np.exp(log_scale)

    times = np.linspace(0.0, float(horizon_years), int(grid_points))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        survival = np.exp(-np.power(times[None, :] / scale[:, None], k[:, None]))
    survival = np.clip(survival, 0.0, 1.0)
    risk = 1.0 - survival

    survival_median = np.nanquantile(survival, 0.5, axis=0)
    survival_low = np.nanquantile(survival, CI_LOW, axis=0)
    survival_high = np.nanquantile(survival, CI_HIGH, axis=0)
    risk_median = np.nanquantile(risk, 0.5, axis=0)
    risk_low = np.nanquantile(risk, CI_LOW, axis=0)
    risk_high = np.nanquantile(risk, CI_HIGH, axis=0)

    median_event_free = scale * np.power(np.log(2.0), 1.0 / k[:n])
    report_years = tuple(y for y in (3, 5, 8, 10, 15) if y <= horizon_years)
    risk_by_year: dict[int, IntervalSummary] = {}
    for year in report_years:
        yearly_risk = 1.0 - np.exp(-np.power(year / scale, k[:n]))
        risk_by_year[year] = _summary(yearly_risk)

    # Clinically useful forecast: probability from the patient's most recent
    # event-free observation, not simply from the original implant date.
    # For a Weibull model this is S(t0 + h) / S(t0). The UI never applies this
    # to profiles whose endpoint has already occurred.
    landmark = max(float(observed_event_free_years or 0.0), 0.0)
    forecast_times = np.linspace(0.0, float(forecast_horizon_years), 121)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        baseline_hazard = np.power(landmark / scale, k[:n])
        future_hazard = np.power(
            (landmark + forecast_times[None, :]) / scale[:, None],
            k[:n, None],
        )
        conditional_event_free = np.exp(-(future_hazard - baseline_hazard[:, None]))
    conditional_event_free = np.clip(conditional_event_free, 0.0, 1.0)
    conditional_event_free_median = np.nanquantile(conditional_event_free, 0.5, axis=0)
    conditional_event_free_low = np.nanquantile(conditional_event_free, CI_LOW, axis=0)
    conditional_event_free_high = np.nanquantile(conditional_event_free, CI_HIGH, axis=0)

    # Approach-matched reference: same posterior approach effect, with no
    # family offset and the labelled-size PPM proxy off.
    reference_scale = np.exp(mu[:n])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        reference_baseline_hazard = np.power(landmark / reference_scale, k[:n])
        reference_future_hazard = np.power(
            (landmark + forecast_times[None, :]) / reference_scale[:, None],
            k[:n, None],
        )
        reference_event_free = np.exp(
            -(reference_future_hazard - reference_baseline_hazard[:, None])
        )
    reference_event_free = np.clip(reference_event_free, 0.0, 1.0)
    reference_event_free_median = np.nanquantile(reference_event_free, 0.5, axis=0)

    conditional_risk_by_horizon: dict[int, IntervalSummary] = {}
    for horizon in (1, 3, 5, 10):
        if horizon > forecast_horizon_years:
            continue
        delta_hazard = (
            np.power((landmark + horizon) / scale, k[:n])
            - np.power(landmark / scale, k[:n])
        )
        conditional_risk_by_horizon[horizon] = _summary(1.0 - np.exp(-delta_hazard))

    # Posterior time from the landmark until conditional survival reaches
    # 50%: solve S(t0 + h) / S(t0) = 0.5 for h in every posterior draw.
    remaining_median = np.power(
        np.power(landmark, k[:n]) + np.log(2.0) * np.power(scale, k[:n]),
        1.0 / k[:n],
    ) - landmark

    if "SAVR" in posterior.approach_effects:
        savr = _flatten(posterior.approach_effects["SAVR"])
        mu_for_ratio, savr = _same_length(mu[:n], savr)
        approach_log_ratio = mu_for_ratio - savr
    else:
        approach_log_ratio = np.zeros(n, dtype=float)

    approach_factor = TimeRatioFactor(
        key="approach",
        label=f"Implant approach: {approach}",
        description="Posterior time ratio relative to the SAVR approach reference.",
        ratio=_factor_summary(approach_log_ratio),
    )

    if family_applied:
        family_factor = TimeRatioFactor(
            key="family",
            label="Valve family",
            description=str(valve_family).replace("_", " "),
            ratio=_factor_summary(family_offset[:n]),
        )
    else:
        family_factor = TimeRatioFactor(
            key="family",
            label="Valve family",
            description=(
                "Unknown or not represented; the model uses the approach-level posterior."
            ),
            ratio=IntervalSummary(1.0, 1.0, 1.0),
            applied=False,
        )

    if size_known:
        ppm_factor = TimeRatioFactor(
            key="ppm",
            label="Small-valve PPM proxy",
            description=(
                "Applied (labelled size <=21 mm)."
                if ppm_proxy_flag
                else "Not triggered (labelled size >21 mm)."
            ),
            ratio=_factor_summary(beta[:n] * float(ppm_proxy_flag)),
            applied=bool(ppm_proxy_flag),
        )
    else:
        ppm_factor = TimeRatioFactor(
            key="ppm",
            label="Small-valve PPM proxy",
            description="Labelled size unavailable; the fitted pipeline defaults this proxy off.",
            ratio=IntervalSummary(1.0, 1.0, 1.0),
            applied=False,
        )

    total_log_ratio = approach_log_ratio[:n] + family_offset[:n] + ppm_term[:n]

    return DurabilityPrediction(
        times=times,
        survival_median=survival_median,
        survival_low=survival_low,
        survival_high=survival_high,
        risk_median=risk_median,
        risk_low=risk_low,
        risk_high=risk_high,
        forecast_times=forecast_times,
        conditional_event_free_median=conditional_event_free_median,
        conditional_event_free_low=conditional_event_free_low,
        conditional_event_free_high=conditional_event_free_high,
        reference_event_free_median=reference_event_free_median,
        conditional_risk_by_horizon=conditional_risk_by_horizon,
        landmark_years=landmark,
        remaining_median_years=_summary(remaining_median),
        median_event_free_years=_summary(median_event_free),
        scale_years=_summary(scale),
        risk_by_year=risk_by_year,
        factors=(approach_factor, family_factor, ppm_factor),
        total_time_ratio=_factor_summary(total_log_ratio),
        posterior_draws=n,
        family_effect_applied=family_applied,
        ppm_effect_applied=ppm_applied,
    )
