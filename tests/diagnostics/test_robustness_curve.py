"""Input stress scenarios preserve held-out rows, metadata and metric weights."""

from dataclasses import replace
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from ml_toolkit.model_evaluation import RobustnessCurve
from ml_toolkit.config import FigureConfig
from ml_toolkit.data import Pool
from ml_toolkit.metrics import MetricScores
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.plots.robustness import robustness_curve
from ml_toolkit.plots.utils import PlotError


def data_pool():
    return Pool(
        X=pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 10.125, 11.875, 12.625, 13.375],
                "constant": [5.0] * 8,
                "cat": ["a", "b"] * 4,
            }
        ),
        y=pl.Series("target", [0.0, 1.0, 2.0, 3.0, 10.125, 11.875, 12.625, 13.375]),
        time=pl.Series(
            "date", [date(2024, 1, 1) + timedelta(days=i) for i in range(8)]
        ),
        weight=pl.Series("weight", [1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 4.0]),
        cat_features=("cat",),
        numerical_features=("x", "constant"),
        object_id=pl.Series("id", list(range(8))),
        auxiliary_features=pl.DataFrame({"group": ["z"] * 8}),
    )


def mae(y, prediction, weight):
    return MetricScores({"mae": float(np.average(abs(y - prediction), weights=weight))})


class RecordingModel:
    def __init__(self):
        self.pools = []

    def predict_pool(self, data):
        self.pools.append(data)
        return data.X["x"].fill_null(0.0).to_numpy().astype(float)


def calculate(report, data=None, model=None, metric=mae, split=None):
    data = data_pool() if data is None else data
    model = RecordingModel() if model is None else model
    split = pl.Series(["train"] * 4 + ["val"] * 4) if split is None else split
    return report.calculate(model=model, metric=metric, data=data, split=split)


def test_missing_preserves_originals_and_rows_and_metrics_use_weights():
    data, model = data_pool(), RecordingModel()
    original = data.X.clone()
    report = RobustnessCurve(features=("x",), levels=(0.0, 1.0), n_repeats=2)
    table = calculate(report, data, model)
    baseline = table.filter(pl.col("is_baseline"))
    assert baseline["value"].to_list() == [0.0]
    assert baseline["level"].to_list() == [None]
    assert baseline["repeat"].to_list() == [None]
    assert table.filter(pl.col("level") == 0.0)["value"].to_list() == [0.0, 0.0]
    expected = float(
        np.average(data.y.tail(4).to_numpy(), weights=data.weight.tail(4).to_numpy())
    )
    assert table.filter(pl.col("level") == 1.0)["value"].to_list() == pytest.approx(
        [expected] * 2
    )
    assert data.X.equals(original)
    for changed in model.pools:
        assert changed.y.equals(data.y.tail(4))
        assert changed.time.equals(data.time.tail(4))
        assert changed.weight.equals(data.weight.tail(4))
        assert changed.object_id.equals(data.object_id.tail(4))
        assert changed.auxiliary_features.equals(data.auxiliary_features.tail(4))
        assert changed.cat_features == data.cat_features
        assert changed.X["cat"].equals(data.X["cat"].tail(4))


def test_noise_uses_weighted_training_scale_and_paired_standard_noise():
    data, model = data_pool(), RecordingModel()
    report = RobustnessCurve(
        features=("x", "constant"),
        mode="noise",
        levels=(0.0, 0.5, 1.0),
        n_repeats=2,
        seed=12,
    )
    table = calculate(report, data, model)
    for repeat in range(2):
        original = model.pools[0].X["x"].to_numpy()
        low = model.pools[2 + 3 * repeat].X["x"].to_numpy() - original
        high = model.pools[3 + 3 * repeat].X["x"].to_numpy() - original
        np.testing.assert_allclose(high, 2 * low)
        rng = np.random.default_rng(np.random.SeedSequence([12, repeat, 1]))
        values = data.X["x"].head(4).to_numpy()
        weight = data.weight.head(4).to_numpy()
        center = np.average(values, weights=weight)
        scale = np.sqrt(np.average((values - center) ** 2, weights=weight))
        np.testing.assert_allclose(high, rng.standard_normal(4) * scale)
    assert table["n_rows"].to_list() == [4] * 7
    assert all(pool.X["constant"].to_list() == [5.0] * 4 for pool in model.pools)
    again = calculate(report, data, RecordingModel())
    assert table.equals(again)


def test_noise_needs_training_rows_and_skips_zero_weight_training_values():
    data = data_pool()
    with pytest.raises(ValueError, match="require train rows"):
        calculate(
            RobustnessCurve(features=("x",), mode="noise"),
            data,
            split=pl.Series(["test"] * 8),
        )
    train = data.X["x"].to_numpy().copy()
    train[0] = 1e9
    altered = replace(
        data,
        X=data.X.with_columns(pl.Series("x", train)),
        weight=pl.Series([0.0, 3.0, 1.0, 3.0, 1.0, 2.0, 3.0, 4.0]),
    )
    model = RecordingModel()
    calculate(
        RobustnessCurve(features=("x",), mode="noise", levels=(1.0,), n_repeats=1),
        altered,
        model,
    )
    expected = np.random.default_rng(np.random.SeedSequence([0, 0, 1])).standard_normal(
        4
    )
    weight = np.array([3.0, 1.0, 3.0])
    values = np.array([1.0, 2.0, 3.0])
    scale = np.sqrt(
        np.average((values - np.average(values, weights=weight)) ** 2, weights=weight)
    )
    np.testing.assert_allclose(
        model.pools[-1].X["x"].to_numpy() - data.X["x"].tail(4).to_numpy(),
        expected * scale,
    )


def test_rounding_is_deterministic_and_uses_decimal_places():
    data, model = data_pool(), RecordingModel()
    table = calculate(
        RobustnessCurve(
            features=("x",), mode="rounding", levels=(2.0, 0.0, 1.0), n_repeats=5
        ),
        data,
        model,
    )
    assert table["repeat"].to_list() == [None, 0, 0, 0]
    for level, changed in zip((0, 1, 2), model.pools[1:]):
        np.testing.assert_allclose(
            changed.X["x"].to_numpy(), np.round(data.X["x"].tail(4).to_numpy(), level)
        )


@pytest.mark.parametrize("dtype", [pl.String, pl.Categorical, pl.Enum(["a", "b"])])
def test_unknown_inserts_unseen_label_without_mutating_category_dictionary(dtype):
    data = data_pool()
    data = replace(data, X=data.X.with_columns(pl.col("cat").cast(dtype)))
    before = data.X.clone()
    model = RecordingModel()
    calculate(
        RobustnessCurve(features=("cat",), mode="unknown", levels=(1.0,), n_repeats=1),
        data,
        model,
    )
    changed = model.pools[-1]
    assert changed.X["cat"].to_list() == ["unknown"] * 4
    assert changed.X["cat"].dtype == pl.String
    assert changed.cat_features == ("cat",)
    assert data.X.equals(before)


def test_unknown_differs_from_missing_and_rejects_existing_sentinel():
    data = data_pool()
    data = replace(data, X=data.X.with_columns(pl.Series("cat", ["unknown", "b"] * 4)))
    with pytest.raises(ValueError, match="must not occur"):
        calculate(RobustnessCurve(features=("cat",), mode="unknown"), data)
    model = RecordingModel()
    calculate(
        RobustnessCurve(
            features=("cat",),
            mode="unknown",
            unknown_value="new",
            levels=(1.0,),
            n_repeats=1,
        ),
        data,
        model,
    )
    assert model.pools[-1].X["cat"].to_list() == ["new"] * 4


def test_sample_rows_are_fixed_across_scenarios():
    data, model = data_pool(), RecordingModel()
    report = RobustnessCurve(
        features=("x",), max_samples=2, levels=(0.0, 0.5, 1.0), n_repeats=2, seed=1
    )
    table = calculate(report, data, model)
    assert table["n_rows"].to_list() == [2] * 7
    ids = model.pools[0].object_id.to_list()
    assert all(pool.object_id.to_list() == ids for pool in model.pools)


@pytest.mark.parametrize("max_samples", [1, None])
def test_zero_weight_rows_are_excluded_before_sampling(max_samples):
    data = replace(
        data_pool(), weight=pl.Series([1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 0.0, 1.0])
    )
    model = RecordingModel()
    report = RobustnessCurve(
        features=("x",),
        levels=(0.0, 1.0),
        n_repeats=1,
        max_samples=max_samples,
        seed=1,
    )
    table = calculate(
        report,
        data,
        model,
        split=pl.Series(["train"] * 6 + ["val"] * 2),
    )
    assert table["n_rows"].to_list() == [1] * 3
    assert table["sum_weight"].to_list() == [1.0] * 3
    assert all(pool.object_id.to_list() == [7] for pool in model.pools)
    assert table.filter(pl.col("is_baseline"))["value"].to_list() == [0.0]


def test_train_only_skips_model_and_artifacts(tmp_path):
    report = RobustnessCurve(features=("x",), mode="noise")
    data, model = data_pool(), RecordingModel()
    table = calculate(report, data, model, split=pl.Series(["train"] * 8))
    assert table.is_empty() and table.schema["level"] == pl.Float64
    assert model.pools == []
    assert (
        report.write_artifacts(
            model=model,
            metric=mae,
            data=data,
            split=pl.Series(["train"] * 8),
            artifact_dir=tmp_path,
            figure=FigureConfig(),
        )
        == []
    )


def test_multiple_metrics_undefined_values_and_one_pass_artifacts(tmp_path):
    data, model = data_pool(), RecordingModel()
    calls = []

    def metric(y, prediction, weight):
        calls.append(1)
        return MetricScores(
            {
                "mae": float(np.average(abs(y - prediction), weights=weight)),
                "undefined": float("nan"),
            }
        )

    report = RobustnessCurve(features=("x",), levels=(0.0, 1.0), n_repeats=2)
    paths = report.write_artifacts(
        model=model,
        metric=metric,
        data=data,
        split=pl.Series(["train"] * 4 + ["test"] * 4),
        artifact_dir=tmp_path,
        figure=FigureConfig(format="svg"),
    )
    assert len(calls) == 5
    table = pl.read_parquet(paths[0])
    assert (
        table.height == 10
        and table.filter(pl.col("metric") == "undefined")["value"].null_count() == 5
    )
    assert [path.name for path in paths] == [
        "metrics.parquet",
        "000_mae.svg",
        "001_undefined.svg",
    ]
    assert all(path.is_file() for path in paths)
    assert not plt.get_fignums()


@pytest.mark.parametrize(
    "settings",
    [
        {"features": ()},
        {"features": ("x", "x")},
        {"features": ("",)},
        {"mode": "other"},
        {"levels": ()},
        {"levels": (0.1, 0.1)},
        {"levels": (float("nan"),)},
        {"levels": (-0.1,)},
        {"levels": (1.1,)},
        {"mode": "rounding", "levels": (0.5,)},
        {"n_repeats": 0},
        {"max_samples": 0},
        {"seed": -1},
        {"unknown_value": ""},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        RobustnessCurve(**{"features": ("x",), **settings})


@pytest.mark.parametrize(
    "report,error",
    [
        (RobustnessCurve(features=("absent",)), "missing from X"),
        (RobustnessCurve(features=("cat",), mode="noise"), "numerical model"),
        (
            RobustnessCurve(features=("cat",), mode="rounding", levels=(0.0,)),
            "numerical model",
        ),
        (RobustnessCurve(features=("x",), mode="unknown"), "categorical model"),
    ],
)
def test_invalid_feature_domains(report, error):
    with pytest.raises(ValueError, match=error):
        calculate(report)


def test_failure_preserves_exception_and_adds_scenario_context():
    error = RuntimeError("prediction failed")

    class FailingModel(RecordingModel):
        def predict_pool(self, data):
            if data.X["x"].null_count():
                raise error
            return super().predict_pool(data)

    with pytest.raises(RuntimeError) as caught:
        calculate(
            RobustnessCurve(features=("x",), levels=(1.0,), n_repeats=1),
            model=FailingModel(),
        )
    assert caught.value is error
    assert any("mode=missing, level=1.0, repeat=0" in note for note in error.__notes__)


def test_catboost_accepts_missing_and_unknown_scenarios():
    data = data_pool()
    model = CatBoostRegressionModel(
        {
            "iterations": 5,
            "depth": 2,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
        }
    ).fit(data.take(range(4)))
    for mode, features in (("missing", ("x",)), ("unknown", ("cat",))):
        table = calculate(
            RobustnessCurve(
                features=features, mode=mode, levels=(0.0, 1.0), n_repeats=1
            ),
            data,
            model,
        )
        assert table["value"].null_count() == 0


def test_lightgbm_accepts_all_scenarios_through_pool_prediction():
    lgb = pytest.importorskip("lightgbm")
    from ml_toolkit.models.lightgbm import LightGBMServingModel

    data = data_pool()
    train = data.take(range(4))
    native = lgb.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 1,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
        },
        lgb.Dataset(train.to_model_input(), label=train.y.to_numpy()),
        num_boost_round=3,
    )
    model = LightGBMServingModel(
        native, numerical_features=["x", "constant"], cat_features=["cat"]
    )
    for mode, features in (
        ("missing", ("x",)),
        ("noise", ("x",)),
        ("rounding", ("x",)),
        ("unknown", ("cat",)),
    ):
        table = calculate(
            RobustnessCurve(
                features=features, mode=mode, levels=(0.0, 1.0), n_repeats=1
            ),
            data,
            model,
        )
        assert table["value"].null_count() == 0


def test_invalid_weights_and_changed_metric_names():
    data = data_pool()
    invalid = replace(data, weight=pl.Series([1.0, 3.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="positive total weight"):
        calculate(RobustnessCurve(features=("x",)), invalid)
    calls = []

    def changed_names(y, prediction, weight):
        calls.append(1)
        return MetricScores({"first" if len(calls) == 1 else "other": 1.0})

    with pytest.raises(ValueError, match="Metric names must match"):
        calculate(
            RobustnessCurve(features=("x",), levels=(1.0,), n_repeats=1),
            metric=changed_names,
        )


def test_plot_errors_close_figures(tmp_path, monkeypatch):
    table = calculate(RobustnessCurve(features=("x",), levels=(0.0, 1.0), n_repeats=1))
    with pytest.raises(PlotError, match="No robustness values"):
        robustness_curve(table, metric="absent", save_fig=tmp_path / "bad.png")

    def fail(*args, **kwargs):
        raise OSError("save failed")

    monkeypatch.setattr("ml_toolkit.plots.robustness.save_figure", fail)
    with pytest.raises(OSError):
        robustness_curve(table, metric="mae", save_fig=tmp_path / "bad.png")
    assert not plt.get_fignums()
