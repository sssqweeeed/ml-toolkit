"""Exposure-weighted regression metrics against independently calculated values."""

import numpy as np
import pytest

from ml_toolkit.metrics import metrics_frequency, metrics_severity


@pytest.mark.parametrize("kind", ["frequency", "severity"])
def test_weighted_errors_deviance_and_null_model_score(kind):
    observed = np.array([1.0, 2.0, 4.0])
    predicted = np.array([2.0, 2.0, 3.0])
    exposure = np.array([1.0, 2.0, 3.0])
    mean = 17 / 6
    if kind == "frequency":
        result = metrics_frequency(observed, predicted, exposure)
        deviance = 2 * (np.log(0.5) + 1 + 3 * (4 * np.log(4 / 3) - 1)) / 6
        null_deviance = 2 * sum(
            w * (y * np.log(y / mean) - y + mean)
            for y, w in zip(observed, exposure)
        ) / 6
        assert result.mean_poisson_deviance == pytest.approx(deviance)
        assert result.d2_poisson_score == pytest.approx(1 - deviance / null_deviance)
    else:
        result = metrics_severity(observed, predicted, exposure, power=2)
        deviance = 2 * (-np.log(0.5) - 0.5 + 3 * (-np.log(4 / 3) + 1 / 3)) / 6
        null_deviance = 2 * sum(
            w * (-np.log(y / mean) + y / mean - 1)
            for y, w in zip(observed, exposure)
        ) / 6
        assert result.mean_tweedie_deviance == pytest.approx(deviance)
        assert result.d2_tweedie_score == pytest.approx(1 - deviance / null_deviance)
    assert result.mean_absolute_error == pytest.approx(4 / 6)
    assert result.root_mean_squared_error == pytest.approx(np.sqrt(4 / 6))
    assert result.prediction_over_observed == pytest.approx(15 / 17)
    assert result.gain_at_10 == pytest.approx(2.4 / 17)
    assert result.lift_at_10 == pytest.approx(24 / 17)
    assert result.to_dict()["prediction_over_observed"] == result.prediction_over_observed
