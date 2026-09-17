# AVR Durability Dashboard

This Streamlit interface presents the repository's two existing heads without
rerunning or changing either model:

- **Head B — current valve state:** reads `labels.csv` and `notes_audit.csv`
  to display HVD/BVF stage, phenotype, confidence, rationale, and traceable
  evidence spans.
- **Head A — future durability:** reads the fitted
  `reports/head_a_posterior.nc` artifact and computes patient-level posterior
  survival trajectories and AFT time-ratio decomposition from the actual
  approach, valve-family, and labelled-size PPM-proxy inputs.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-ui.txt
streamlit run app.py
```

The application expects the committed pipeline artifacts at their current
repository paths. It does not need PyMC to run; ArviZ reads the existing
posterior NetCDF file.

Run the focused UI checks with:

```bash
python -m pytest -q tests/test_data.py tests/test_predictor.py tests/test_reporting.py
```

## UI contract

The presentation layer depends only on:

| Artifact | Purpose |
|---|---|
| `labels.csv` | Head B patient state, rationale, and evidence |
| `notes_audit.csv` | Per-patient extraction audit counts |
| `data_processed_patient_features.csv` | Head A covariates |
| `data_processed_patient_labels.csv` | Event/censoring metadata |
| `reports/head_a_posterior.nc` | Posterior draws for prediction |
| `reports/head_a_bootstrap_bayesian.yaml` | Validation summary |

Model-specific logic is isolated in `ui/predictor.py`; Streamlit rendering is
kept in `app.py`. This separation is intentional so later changes from
`paschalis-dev` can be merged with a small, explicit adapter update rather than
rewriting the interface.

## Deliberate scientific boundaries

- The current Head A does not model competing all-cause mortality. The UI never
  labels `1 - S(t)` as a cumulative-incidence function.
- SHAP is not shown as the primary explanation because the repository's SHAP
  output belongs to the XGBoost comparator, not the Bayesian model presented in
  the dashboard.
- No age, CKD, diabetes, smoking, or sparse echo fields are exposed as model
  inputs because they are not covariates in the fitted Head A.
- The interface does not prescribe surveillance intervals or treatment.
- Every screen and export is labelled as a research prototype not validated for
  clinical use.
