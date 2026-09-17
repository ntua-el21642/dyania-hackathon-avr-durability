import numpy as np

from ui.predictor import PosteriorDraws, predict_durability


def posterior_fixture() -> PosteriorDraws:
    draws = 400
    return PosteriorDraws(
        shape_k=np.full(draws, 2.0),
        approach_effects={
            "SAVR": np.full(draws, np.log(10.0)),
            "TAVR": np.full(draws, np.log(12.0)),
        },
        family_offsets={
            "known_family": np.full(draws, np.log(1.1)),
        },
        beta_ppm=np.full(draws, np.log(0.8)),
    )


def test_prediction_curve_is_bounded_and_monotonic():
    result = predict_durability(
        posterior_fixture(),
        approach="SAVR",
        valve_family="known_family",
        family_known=True,
        ppm_proxy_flag=False,
        size_known=True,
        horizon_years=15,
    )

    assert np.all((result.survival_median >= 0) & (result.survival_median <= 1))
    assert np.all(np.diff(result.survival_median) <= 1e-12)
    assert np.all(np.diff(result.risk_median) >= -1e-12)
    assert np.allclose(result.risk_median, 1 - result.survival_median)
    assert result.risk_by_year[5].median > 0
    assert np.all((result.reference_event_free_median >= 0) & (result.reference_event_free_median <= 1))


def test_conditional_forecast_starts_at_one_and_declines():
    result = predict_durability(
        posterior_fixture(),
        approach="SAVR",
        valve_family="known_family",
        family_known=True,
        ppm_proxy_flag=True,
        size_known=True,
        observed_event_free_years=6.0,
    )

    assert np.isclose(result.conditional_event_free_median[0], 1.0)
    assert np.isclose(result.reference_event_free_median[0], 1.0)
    assert np.all(np.diff(result.conditional_event_free_median) <= 1e-12)
    assert result.remaining_median_years.median > 0
    assert result.remaining_median_years.median < result.median_event_free_years.median


def test_time_ratio_decomposition_matches_aft_parameterization():
    result = predict_durability(
        posterior_fixture(),
        approach="TAVR",
        valve_family="known_family",
        family_known=True,
        ppm_proxy_flag=True,
        size_known=True,
    )

    # TAVR/SAVR = 1.2, family = 1.1, PPM term = 0.8.
    assert np.isclose(result.factors[0].ratio.median, 1.2)
    assert np.isclose(result.factors[1].ratio.median, 1.1)
    assert np.isclose(result.factors[2].ratio.median, 0.8)
    assert np.isclose(result.total_time_ratio.median, 1.2 * 1.1 * 0.8)


def test_unknown_family_and_size_are_explicit_reference_fallbacks():
    result = predict_durability(
        posterior_fixture(),
        approach="SAVR",
        valve_family=None,
        family_known=False,
        ppm_proxy_flag=False,
        size_known=False,
    )

    assert not result.family_effect_applied
    assert not result.ppm_effect_applied
    assert result.factors[1].ratio.median == 1.0
    assert result.factors[2].ratio.median == 1.0
    assert "unavailable" in result.factors[2].description.lower()
