# Guyot KM-Curve Digitization Checklist (Master Prompt Section 5.6)

## Finding: no usable source figure exists in the 3 provided PDFs

All three provided papers (`papers/ejcts_52_3_408.pdf`, `papers/JAH3-14-e041505.pdf`, `papers/ehaa799.pdf`) were
searched full-text on 2026-09-17 for Kaplan-Meier content (`"kaplan"`, `"at risk"`, `"figure"`, near-match
context). Result:

- **`ejcts_52_3_408.pdf` (Capodanno 2017) Figure 4** is the only figure-level KM-related content — but it is
  explicitly captioned "**Schematic** representation of the effect of using actual vs. actuarial [methods]," i.e.
  an illustrative diagram, not a real patient-data curve with a numbers-at-risk table. Not usable.
- **`JAH3-14-e041505.pdf` (Trimaille 2025)** and **`ehaa799.pdf` (VARC-3)** mention Kaplan-Meier only in
  methodological/reporting-standard prose (e.g. "mortality should be reported using Kaplan-Meier methods") — no
  plotted curve, no at-risk table, anywhere in either paper.
- All three papers' actual durability data (the numbers used to fit `config/priors.yaml`) come from **summary
  statistics in tables** (Capodanno Table 1/2; Trimaille's pooled table), not from digitizable curves.

**Consequence:** the Guyot reconstruction cannot proceed with the files currently provided. Per Hard Rule #1 (never
fabricate data or coordinates) and the Master Prompt's own instruction ("if digitization has not been done by a
human, stop and produce a digitization checklist instead"), no reconstruction was attempted. `src/external/guyot.py`
implements and self-tests the algorithm against synthetic data (see below) so the method is ready to run the moment
real digitized input exists — it has simply never been pointed at real data.

## What a team member would need to source

The papers we have are **reviews/consensus statements that cite primary studies in tables**, not the primary
studies themselves. To digitize a real KM curve, a team member needs direct access to the primary papers listed
below (all cited in `ejcts_52_3_408.pdf` Table 1/2 or `JAH3-14-e041505.pdf`'s table) and to check each for an
actual plotted KM figure with a numbers-at-risk row:

| Priority | Study | Valve family | Why this one first |
|---|---|---|---|
| 1 | Bourguignon et al. 2015 (Ann Thorac Surg) | Carpentier-Edwards Perimount | Largest SAVR-pericardial cohort (n=2758) informing our `carpentier_edwards_savr_pericardial` prior; a real curve would let us replace the current 2-point least-squares fit with a full reconstruction. |
| 2 | Pibarot et al. 2020 / Mack et al. 2023 | Sapien XT / Sapien 3 | Directly informs `sapien_family_tavr_balloon`, currently a single pooled 5-year point only. |
| 3 | Alaour et al. 2025 | Mixed TAVR (VARC-3) | 10-year anchor for `tavr_general_population`; largest available TAVR follow-up horizon in our source set. |
| 4 | Mohammadi et al. 2012 | Freestyle | Already 2-point (10y/15y) in our current fit; a full curve would materially sharpen the `freestyle_savr_stentless` shape estimate specifically. |
| 5 | David et al. 2010 | Hancock II | Single-point family; highest-value target for converting a borrowed-shape prior into a real fitted one. |

## Procedure once a candidate figure is obtained (for whoever picks this up)

1. Confirm the figure reports numbers-at-risk at regular intervals below the x-axis (full Guyot) — if not, check
   whether total N and total events are reported anywhere in the paper (reduced Guyot variant, Section 5.6).
2. Digitize curve coordinates with WebPlotDigitizer (or equivalent); export as `(time, survival)` pairs.
3. Save to `data/external/km_digitized/<study>_<endpoint>.csv` with columns `time_years, survival_pct`.
4. Save a metadata YAML alongside it: `paper`, `figure`, `endpoint definition` (verbatim from the paper), `population`
   (age range, risk profile), `valve family`, `n at risk table` (if available).
5. Run `python src/external/guyot.py --curve <csv> --meta <yaml>` (implemented and self-tested against synthetic
   data — see `reports/guyot_selftest.yaml` — never run against real data yet).
6. Compare the reconstructed KM curve against the digitized original; report the maximum absolute deviation
   (the self-test script already implements this comparison logic against a synthetic target).
7. Use the resulting pseudo-IPD for baseline-hazard-only priors per valve family (no covariates), replacing the
   corresponding single/2-point least-squares fit in `config/priors.yaml`, with old and new versions both retained
   for a documented before/after comparison.

**Not attempted beyond this document and the self-tested algorithm implementation** — no real coordinates exist to
run it on.
