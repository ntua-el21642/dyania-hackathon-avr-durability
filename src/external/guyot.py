"""
Guyot et al. (BMC Med Res Methodol 2012) Kaplan-Meier curve reconstruction:
turns digitized (time, survival) curve points + a numbers-at-risk table
into pseudo individual patient data (event/censoring times), so a baseline
hazard per valve family can be estimated from a published curve rather
than the handful of single/two-point summary statistics currently used in
`config/priors.yaml`.

**Never run against real data in this build** — see
`data/external/km_digitized/DIGITIZATION_CHECKLIST.md`: none of the three
provided source PDFs contain an actual plotted KM curve with a
numbers-at-risk table (only a schematic figure and reporting-standard
prose). Self-tested here against SYNTHETIC data only, to demonstrate the
method is implemented correctly and is ready to run the moment real
digitized coordinates exist.

Simplified relative to the full published algorithm (documented, not
hidden): within each at-risk interval, events are apportioned to curve
steps by the survival-drop ratio and rounded to the nearest integer count
(the original paper's fractional-event/optimization refinement for exact
total-event-count matching is not implemented); remaining censoring within
an interval is placed at the interval's right edge rather than distributed
across it. Adequate for a baseline-hazard prior, not claimed to be
publication-grade for a primary analysis.
"""
from __future__ import annotations

import numpy as np


def reconstruct_ipd(curve_t: np.ndarray, curve_s: np.ndarray,
                     risk_t: np.ndarray, risk_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (times, event) pseudo-IPD arrays. curve_t/curve_s: digitized
    (time, survival) points, time-sorted, survival non-increasing, survival
    at t=0 assumed 1.0 if not explicitly given. risk_t/risk_n: at-risk
    table time points and counts, risk_t[0] should be 0 (or the first
    curve time) with risk_n[0] = total N."""
    curve_t = np.asarray(curve_t, dtype=float)
    curve_s = np.asarray(curve_s, dtype=float)
    risk_t = np.asarray(risk_t, dtype=float)
    risk_n = np.asarray(risk_n, dtype=float)

    times, events = [], []
    n_cur = risk_n[0]

    for j in range(len(risk_t) - 1):
        t_start, t_end = risk_t[j], risk_t[j + 1]
        n_end_target = risk_n[j + 1]

        mask = (curve_t > t_start) & (curve_t <= t_end)
        seg_t, seg_s = curve_t[mask], curve_s[mask]

        s_prev = np.interp(t_start, curve_t, curve_s, left=1.0)
        for t_i, s_i in zip(seg_t, seg_s):
            if s_prev <= 0 or n_cur <= 0:
                break
            d_i = int(round(n_cur * (1 - s_i / s_prev)))
            d_i = max(0, min(d_i, int(n_cur)))
            for _ in range(d_i):
                times.append(t_i)
                events.append(1)
            n_cur -= d_i
            s_prev = s_i

        # remaining discrepancy this interval -> censored, placed at interval end
        c_j = max(0, int(round(n_cur - n_end_target)))
        for _ in range(c_j):
            times.append(t_end)
            events.append(0)
        n_cur -= c_j
        n_cur = max(n_cur, n_end_target)  # reconcile to the reported at-risk count

    # anyone left at the last risk-table point is censored there
    if n_cur > 0:
        for _ in range(int(round(n_cur))):
            times.append(risk_t[-1])
            events.append(0)

    return np.array(times), np.array(events)


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard KM estimator from IPD -> (unique event times, survival)."""
    order = np.argsort(times)
    times, events = times[order], events[order]
    unique_t = np.unique(times[events == 1])
    s = 1.0
    surv = []
    for t in unique_t:
        n_at_risk = int((times >= t).sum())
        d = int(((times == t) & (events == 1)).sum())
        if n_at_risk > 0:
            s *= (1 - d / n_at_risk)
        surv.append(s)
    return unique_t, np.array(surv)


def max_abs_deviation(recon_t, recon_s, target_t, target_s) -> float:
    grid = np.union1d(recon_t, target_t)
    r = np.interp(grid, recon_t, recon_s, left=1.0, right=recon_s[-1] if len(recon_s) else 1.0)
    tg = np.interp(grid, target_t, target_s, left=1.0, right=target_s[-1] if len(target_s) else 1.0)
    return float(np.max(np.abs(r - tg)))


def _self_test(seed: int = 20260917, n: int = 200) -> dict:
    """Synthetic round-trip test: simulate n patients from a known Weibull,
    compute their TRUE KM curve + at-risk table (playing the role of
    'digitized data'), run reconstruct_ipd, and check the reconstructed KM
    matches the true curve closely."""
    rng = np.random.default_rng(seed)
    true_k, true_scale = 2.0, 10.0
    event_t = true_scale * (-np.log(rng.uniform(1e-6, 1, n))) ** (1 / true_k)
    censor_t = rng.uniform(0, 15, n)
    obs_t = np.minimum(event_t, censor_t)
    obs_e = (event_t <= censor_t).astype(int)

    true_curve_t, true_curve_s = kaplan_meier(obs_t, obs_e)
    true_curve_t = np.concatenate([[0.0], true_curve_t])
    true_curve_s = np.concatenate([[1.0], true_curve_s])

    risk_t = np.arange(0, 16, 1.0)
    risk_n = np.array([int((obs_t >= t).sum()) for t in risk_t], dtype=float)
    risk_n[0] = n

    recon_t_ipd, recon_e_ipd = reconstruct_ipd(true_curve_t, true_curve_s, risk_t, risk_n)
    recon_km_t, recon_km_s = kaplan_meier(recon_t_ipd, recon_e_ipd)
    recon_km_t = np.concatenate([[0.0], recon_km_t])
    recon_km_s = np.concatenate([[1.0], recon_km_s])

    dev = max_abs_deviation(recon_km_t, recon_km_s, true_curve_t, true_curve_s)
    return dict(
        n_simulated=n, n_reconstructed_events=int(recon_e_ipd.sum()), n_true_events=int(obs_e.sum()),
        max_abs_deviation=dev,
        pass_threshold_0_10=bool(dev < 0.10),
        note="Synthetic self-test only -- see module docstring and DIGITIZATION_CHECKLIST.md; "
             "never run against real published data in this build.",
    )


if __name__ == "__main__":
    import yaml
    result = _self_test()
    print(result)
    with open("reports/guyot_selftest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print("wrote reports/guyot_selftest.yaml")
