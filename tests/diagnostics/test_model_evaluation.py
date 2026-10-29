"""Shrink sampling, metric semantics, errors and exported curve values."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import FigureConfig
from ml_toolkit.metrics import MetricScores
from ml_toolkit.model_evaluation import ModelEvaluator, ShrinkCurve
from ml_toolkit.plots.model_evaluation import shrink_curve
from ml_toolkit.plots.utils import PlotError
from tests.support.training import MeanModel, PrefixModel, make_data, mae


@pytest.mark.parametrize(
    "report,size,expected",
    [
        (ShrinkCurve(), 1, [1]),
        (ShrinkCurve(), 4, [1, 2, 3, 4]),
        (ShrinkCurve(max_points=3), 10, [1, 5, 10]),
        (ShrinkCurve(step=3), 8, [1, 3, 6, 8]),
        (ShrinkCurve(step=100), 8, [1, 8]),
        (ShrinkCurve(step=1), 4, [1, 2, 3, 4]),
        (ShrinkCurve(sizes=(6, 2)), 8, [2, 6, 8]),
    ],
)
def test_sizes_use_fitted_count_and_always_include_full_model(report, size, expected):
    data = make_data()
    model = PrefixModel({"iterations": size}).fit(data)
    before = model.predict_pool(data)
    table = report.calculate(
        model=model, metric=mae, data=data, split=pl.Series(["train"] * len(data))
    )
    assert table["n_iterations"].to_list() == expected
    np.testing.assert_array_equal(before, model.predict_pool(data))
    assert model.n_iterations == size
    assert (
        table["value"][-1]
        == mae(data.y.to_numpy(), before, data.weight.to_numpy()).to_dict()["mae"]
    )


def test_default_limits_work_for_thousands_of_iterations():
    data = make_data()
    table = ShrinkCurve().calculate(
        model=PrefixModel({"iterations": 3000}).fit(data),
        metric=mae,
        data=data,
        split=pl.Series(["val"] * len(data)),
    )
    sizes = table["n_iterations"].to_list()
    assert len(sizes) == 50 and sizes[0] == 1 and sizes[-1] == 3000
    assert sizes == sorted(set(sizes))


@pytest.mark.parametrize(
    "settings",
    [
        {"max_points": 1},
        {"max_points": True},
        {"step": 0},
        {"step": 1.5},
        {"sizes": ()},
        {"sizes": (1, 1)},
        {"sizes": (False,)},
        {"sizes": (0,)},
        {"sizes": [1, 2]},
        {"step": 1, "max_points": 10},
        {"step": 1, "sizes": (1,)},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        ShrinkCurve(**settings)


def test_explicit_sizes_and_missing_capability_fail_before_predictions(monkeypatch):
    data = make_data()
    inputs = dict(metric=mae, data=data, split=pl.Series(["train"] * len(data)))
    model = PrefixModel({"iterations": 3}).fit(data)
    monkeypatch.setattr(model, "predict_pool", lambda data: pytest.fail("predicted"))
    with pytest.raises(ValueError, match="fitted model's 3"):
        ShrinkCurve(sizes=(4,)).calculate(model=model, **inputs)
    with pytest.raises(TypeError, match="MeanModel.*implement.*n_iterations.*shrink"):
        ShrinkCurve().calculate(model=MeanModel({}).fit(data), **inputs)


def test_weighted_metrics_preserve_interleaved_splits_and_undefined_values(tmp_path):
    source = make_data()
    data = replace(source, weight=pl.Series(np.arange(1, len(source) + 1)))
    model = PrefixModel({"iterations": 4}).fit(data)
    split = pl.Series(["val", "train"] * 12)
    calls = []

    def metric(y, prediction, weight):
        calls.append(len(y))
        return MetricScores(
            {
                "absolute/error": float(
                    np.average(abs(y - prediction), weights=weight)
                ),
                "undefined": float("nan"),
                "infinite": float("inf"),
            }
        )

    report = ShrinkCurve(sizes=(2,))
    table = report.calculate(model=model, metric=metric, data=data, split=split)
    assert len(calls) == 4
    assert table.schema == {
        "n_iterations": pl.Int64,
        "split": pl.String,
        "metric": pl.String,
        "value": pl.Float64,
    }
    assert table.filter(pl.col("metric") != "absolute/error")["value"].null_count() == 8
    for row in table.filter(pl.col("metric") == "absolute/error").iter_rows(named=True):
        pool = data.take(np.flatnonzero((split == row["split"]).to_numpy()))
        expected = metric(
            pool.y.to_numpy(),
            model.shrink(row["n_iterations"]).predict_pool(pool),
            pool.weight.to_numpy(),
        )
        assert row["value"] == pytest.approx(expected.to_dict()["absolute/error"])
    calls.clear()
    assert report.evaluate(model=model, metric=metric, data=data, split=split) == {}
    paths = report.write_artifacts(
        model=model,
        metric=metric,
        data=data,
        split=split,
        artifact_dir=tmp_path,
        figure=FigureConfig(format="svg"),
    )
    assert len(calls) == 4  # One calculation, independent of the metric count.
    assert paths[0].name == "metrics.parquet"
    assert pl.read_parquet(paths[0]).equals(table)
    assert [p.name for p in paths[1:]] == [
        "000_absolute_error.svg",
        "001_undefined.svg",
        "002_infinite.svg",
    ]
    assert all(p.is_file() for p in paths)
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "bad_split",
    [pl.Series(["other"] * 24), pl.Series(["train"]), pl.Series(["val", "test"] * 12)],
)
def test_invalid_split(bad_split):
    data = make_data()
    with pytest.raises(ValueError, match="split"):
        ShrinkCurve().calculate(
            model=PrefixModel({}).fit(data), metric=mae, data=data, split=bad_split
        )


def test_prediction_and_metric_errors_keep_iteration_context(monkeypatch):
    data = make_data()
    model = PrefixModel({"iterations": 2}).fit(data)
    inputs = dict(model=model, data=data, split=pl.Series(["train"] * len(data)))
    error = RuntimeError("metric failed")

    def fail(*args):
        raise error

    with pytest.raises(RuntimeError) as caught:
        ShrinkCurve(name="small").calculate(metric=fail, **inputs)
    assert caught.value is error
    assert "evaluator=small, n_iterations=1" in error.__notes__[0]
    monkeypatch.setattr(
        PrefixModel, "predict_pool", lambda self, data: np.zeros((len(data), 2))
    )
    with pytest.raises(ValueError, match="one finite value"):
        ShrinkCurve().calculate(metric=mae, **inputs)


def test_direct_plot_closes_on_save_failure(tmp_path, monkeypatch):
    table = pl.DataFrame(
        {
            "n_iterations": [1, 2],
            "split": ["val"] * 2,
            "metric": ["mae"] * 2,
            "value": [1.0, None],
        }
    )
    with pytest.raises(PlotError, match="No shrink curve"):
        shrink_curve(table, metric="other", save_fig=tmp_path / "missing.png")
    monkeypatch.setattr(
        "ml_toolkit.plots.model_evaluation.save_figure",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("failed")),
    )
    with pytest.raises(OSError):
        shrink_curve(table, metric="mae", save_fig=tmp_path / "curve.png")
    assert not plt.get_fignums()


def test_base_evaluator_has_optional_methods():
    evaluator = ModelEvaluator(name="audit")
    assert evaluator.parameters() == {}
    with pytest.raises(ValueError, match="Evaluator name"):
        ModelEvaluator(name="../bad")
