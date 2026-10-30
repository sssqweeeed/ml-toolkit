"""Weighted residual mathematics, original rows, and report artifacts."""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal, localcontext

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

from ml_toolkit.evaluation import ResidualResult, Residuals
from ml_toolkit.config import FigureConfig
from ml_toolkit.data import Pool
from ml_toolkit.plots.residuals import residual_diagnostics


@pytest.fixture
def inputs():
    data = Pool(
        X=pl.DataFrame({"x": np.arange(8, dtype=float)}),
        y=pl.Series([2.0, 1.0, 100.0, 4.0, 2.0, 7.0, 3.0, 8.0]),
        time=pl.Series([date(2025, 1, 1) + timedelta(days=i) for i in range(8)]),
        weight=pl.Series([1.0, 3.0, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0]),
        object_id=pl.Series("id", ["a", "b", "c", "d", "e", "f", "g", "h"]),
        auxiliary_features=pl.DataFrame({"region": ["east", "west"] * 4}),
    )
    return dict(
        data=data,
        prediction=np.asarray([1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]),
        split=pl.Series(["train"] * 4 + ["test"] * 4),
    )


def test_raw_losses_weighted_quantiles_and_largest_contributors(inputs):
    report = Residuals(n_bins=1, max_rows=2)
    result = report.calculate(**inputs)
    y = inputs["data"].y.to_numpy()
    prediction = inputs["prediction"]
    weight = inputs["data"].weight.to_numpy()
    expected = prediction - y
    np.testing.assert_array_equal(result.rows["residual"].to_numpy(), expected)
    np.testing.assert_array_equal(result.rows["loss"].to_numpy(), expected**2)
    np.testing.assert_array_equal(
        result.rows["contribution"].to_numpy(), expected**2 * weight
    )
    for part in ("train", "test"):
        mask = (inputs["split"] == part).to_numpy() & (weight > 0)
        summary = result.bins.filter(pl.col("split") == part).row(0, named=True)
        assert summary["sum_weight"] == weight[mask].sum()
        assert summary["n_rows"] == mask.sum()
        assert summary["mean_residual"] == pytest.approx(
            np.average(expected[mask], weights=weight[mask])
        )
        expanded = np.repeat(expected[mask], weight[mask].astype(int))
        np.testing.assert_array_equal(
            [summary[name] for name in ("p05", "p25", "p50", "p75", "p95")],
            np.quantile(expanded, [0.05, 0.25, 0.5, 0.75, 0.95], method="inverted_cdf"),
        )
    assert result.worst.filter(pl.col("split") == "train")["row_index"].to_list() == [
        3,
        1,
    ]
    assert result.worst.filter(pl.col("split") == "test")["row_index"].to_list() == [
        5,
        4,
    ]
    assert result.rows["row_index"].to_list() == list(range(8))
    assert result.rows["object_id"].equals(inputs["data"].object_id.rename("object_id"))
    assert result.rows["time"].equals(inputs["data"].time.rename("time"))


def test_absolute_contributions_and_deterministic_ties(inputs):
    result = Residuals(loss="absolute", max_rows=3).calculate(**inputs)
    np.testing.assert_array_equal(
        result.rows["loss"], abs(inputs["prediction"] - inputs["data"].y.to_numpy())
    )
    data = replace(
        inputs["data"],
        y=pl.Series(inputs["prediction"] - 1),
        weight=pl.Series([1.0] * 8),
    )
    tied = Residuals(max_rows=2).calculate(**{**inputs, "data": data})
    assert tied.worst["row_index"].to_list() == [0, 1, 4, 5]


@pytest.mark.parametrize(
    "loss,metric", [("poisson", mean_poisson_deviance), ("gamma", mean_gamma_deviance)]
)
def test_unit_deviance_matches_independent_metric(inputs, loss, metric):
    result = Residuals(loss=loss).calculate(**inputs)
    observed = inputs["data"].y.to_numpy()
    prediction = inputs["prediction"]
    for i in range(len(observed)):
        expected = metric([observed[i]], [prediction[i]])
        assert result.rows["loss"][i] == pytest.approx(expected, abs=1e-12)
        assert result.rows["residual"][i] ** 2 == pytest.approx(expected, abs=1e-12)
        assert np.sign(result.rows["residual"][i]) == np.sign(
            prediction[i] - observed[i]
        )


def test_poisson_zero_target_limit_and_perfect_fit(inputs):
    data = replace(
        inputs["data"], y=pl.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    )
    prediction = np.asarray([2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = Residuals(loss="poisson").calculate(
        **{**inputs, "data": data, "prediction": prediction}
    )
    assert result.rows["loss"][0] == 4.0
    assert result.rows["residual"][0] == 2.0
    assert result.rows["residual"].tail(7).to_list() == [0.0] * 7


def test_deviance_near_perfect_predictions_remains_finite(inputs):
    observed = np.full(8, 1e9)
    prediction = observed + np.arange(8) * 1e-4
    data = replace(inputs["data"], y=pl.Series(observed))
    for loss in ("poisson", "gamma"):
        result = Residuals(loss=loss).calculate(
            **{**inputs, "data": data, "prediction": prediction}
        )
        assert np.isfinite(result.rows["residual"].to_numpy()).all()
        assert (result.rows["loss"] >= 0).all()
        assert result.rows["residual"][0] == 0.0


@pytest.mark.parametrize("loss", ["poisson", "gamma"])
@pytest.mark.parametrize("scale", [1e16, 1e-16, 1e200, 1e-200])
def test_near_perfect_deviance_matches_high_precision_reference(inputs, loss, scale):
    observed = np.full(8, scale)
    prediction = np.asarray(
        [
            scale,
            np.nextafter(scale, np.inf),
            np.nextafter(scale, 0),
            scale * (1 + 1e-8),
            scale * (1 - 1e-8),
            scale * (1 + 1e-4),
            scale * (1 - 1e-4),
            scale * (1 + 5e-4),
        ]
    )
    data = replace(inputs["data"], y=pl.Series(observed))
    result = Residuals(loss=loss).calculate(
        **{**inputs, "data": data, "prediction": prediction}
    )
    with localcontext() as context:
        context.prec = 110
        for i, value in enumerate(prediction):
            y, mu = Decimal.from_float(scale), Decimal.from_float(float(value))
            ratio = y / mu
            if loss == "poisson":
                reference = 2 * (y * ratio.ln() + (mu - y))
            else:
                reference = 2 * (ratio - 1 - ratio.ln())
            expected = float(reference)
            expected_residual = float(reference.sqrt()) * np.sign(value - scale)
            assert result.rows["loss"][i] == pytest.approx(expected, rel=5e-15, abs=0)
            assert result.rows["residual"][i] == pytest.approx(
                expected_residual, rel=5e-15, abs=0
            )


def test_prediction_bins_use_train_weights_and_ignore_test_outliers(inputs):
    report = Residuals(n_bins=3)
    result = report.calculate(**inputs)
    changed = {
        **inputs,
        "prediction": np.concatenate([inputs["prediction"][:4], [1e3, 2e3, 3e3, 4e3]]),
    }
    second = report.calculate(**changed)
    assert_frame_equal(
        result.bins.filter(pl.col("split") == "train"),
        second.bins.filter(pl.col("split") == "train"),
    )
    bounds = result.bins.filter(pl.col("split") == "train")["upper_bound"].to_list()
    assert bounds == [3.0, float("inf")]
    assert second.rows["prediction_bin"].tail(4).to_list() == [1] * 4


def test_constant_predictions_train_only_and_heldout_only(inputs):
    for part in ("train", "test"):
        result = Residuals(n_bins=20).calculate(
            **{
                **inputs,
                "prediction": np.full(8, 3.0),
                "split": pl.Series([part] * 8),
            }
        )
        assert result.bins.height == 1
        assert result.bins["prediction_bin"].to_list() == [0]
        assert result.bins["n_rows"].to_list() == [6]


def test_zero_weight_split_is_retained_but_not_summarized(inputs):
    data = replace(inputs["data"], weight=pl.Series([0.0] * 4 + [1.0] * 4))
    result = Residuals().calculate(**{**inputs, "data": data})
    assert result.rows.height == 8
    assert result.bins["split"].unique().to_list() == ["test"]
    assert result.worst["split"].unique().to_list() == ["test"]


@pytest.mark.parametrize(
    "loss,bad_y,bad_prediction,message",
    [
        ("poisson", -1.0, 1.0, "nonnegative targets"),
        ("poisson", 1.0, 0.0, "positive predictions"),
        ("gamma", 0.0, 1.0, "positive targets"),
        ("gamma", 1.0, -1.0, "positive predictions"),
    ],
)
def test_deviance_domains(inputs, loss, bad_y, bad_prediction, message):
    data = replace(inputs["data"], y=pl.Series([bad_y] * 8))
    with pytest.raises(ValueError, match=message):
        Residuals(loss=loss).calculate(
            **{**inputs, "data": data, "prediction": np.full(8, bad_prediction)}
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("weight", [0.0] * 8, "positive total"),
        ("weight", [-1.0] + [1.0] * 7, "nonnegative"),
        ("weight", [float("nan")] + [1.0] * 7, "finite"),
        ("y", [float("inf")] + [1.0] * 7, "Targets must be finite"),
    ],
)
def test_invalid_observations(inputs, field, value, message):
    data = replace(inputs["data"], **{field: pl.Series(value)})
    with pytest.raises(ValueError, match=message):
        Residuals().calculate(**{**inputs, "data": data})


@pytest.mark.parametrize(
    "settings",
    [
        {"loss": "other"},
        {"n_bins": 0},
        {"histogram_bins": 0},
        {"max_samples": 0},
        {"max_rows": 0},
        {"seed": -1},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        Residuals(**settings)


def test_prediction_and_split_validation(inputs):
    with pytest.raises(ValueError, match="one finite value"):
        Residuals().calculate(**{**inputs, "prediction": np.ones((8, 2))})
    with pytest.raises(ValueError, match="one finite value"):
        Residuals().calculate(**{**inputs, "prediction": np.full(8, np.nan)})
    with pytest.raises(ValueError, match="split"):
        Residuals().calculate(**{**inputs, "split": pl.Series(["train"] * 3)})


def test_inputs_are_unchanged_and_settings_reusable(inputs):
    report = Residuals()
    data = inputs["data"]
    before_frame, before_aux = data.X.clone(), data.auxiliary_features.clone()
    before_prediction, before_split = (
        inputs["prediction"].copy(),
        inputs["split"].clone(),
    )
    first = report.calculate(**inputs)
    second = report.calculate(**inputs)
    assert_frame_equal(first.rows, second.rows)
    assert_frame_equal(first.bins, second.bins)
    assert_frame_equal(data.X, before_frame)
    assert_frame_equal(data.auxiliary_features, before_aux)
    np.testing.assert_array_equal(inputs["prediction"], before_prediction)
    assert inputs["split"].equals(before_split)
    assert report.evaluate(**inputs) == {}


def test_artifacts_calculate_once_and_match_complete_tables(
    inputs, tmp_path, monkeypatch
):
    report = Residuals(max_samples=1)
    expected = report.calculate(**inputs)
    calls = []
    calculate = Residuals.calculate

    def capture(self, **kwargs):
        calls.append(1)
        return calculate(self, **kwargs)

    monkeypatch.setattr(Residuals, "calculate", capture)
    paths = report.write_artifacts(
        **inputs,
        artifact_dir=tmp_path,
        figure=FigureConfig(format="svg"),
        features=None,
        n_bins=2
    )
    assert len(calls) == 1
    assert [path.name for path in paths] == [
        "rows.parquet",
        "bins.parquet",
        "worst.parquet",
        "residuals.svg",
    ]
    for path, table in zip(
        paths[:3], (expected.rows, expected.bins, expected.worst), strict=True
    ):
        assert_frame_equal(pl.read_parquet(path), table)
    assert all(path.is_file() for path in paths)
    assert not plt.get_fignums()


def test_histogram_weights_and_scatter_bound_are_independent_of_tables(
    inputs, tmp_path, monkeypatch
):
    from matplotlib.axes import Axes

    result = Residuals().calculate(**inputs)
    histograms, scatters = [], []
    hist, scatter = Axes.hist, Axes.scatter

    def capture_hist(self, values, *args, **kwargs):
        histograms.append(kwargs["weights"].copy())
        return hist(self, values, *args, **kwargs)

    def capture_scatter(self, x, y, *args, **kwargs):
        scatters.append((len(x), len(y)))
        return scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture_hist)
    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    residual_diagnostics(result, save_fig=tmp_path / "plot.png", max_samples=2)
    assert len(histograms) == 2 and all(
        sum(w) == pytest.approx(1.0) for w in histograms
    )
    assert scatters == [(2, 2), (2, 2)]
    assert result.rows.height == 8


def test_plot_closes_after_save_failure(inputs, tmp_path, monkeypatch):
    result = Residuals().calculate(**inputs)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("ml_toolkit.plots.residuals.save_figure", fail)
    with pytest.raises(OSError, match="disk full"):
        residual_diagnostics(result, save_fig=tmp_path / "plot.png")
    assert not plt.get_fignums()


def test_result_without_object_ids_preserves_clean_schema(inputs):
    data = replace(inputs["data"], object_id=None)
    result = Residuals().calculate(**{**inputs, "data": data})
    assert isinstance(result, ResidualResult)
    assert "object_id" not in result.rows.columns
    assert result.rows.schema == result.worst.schema
