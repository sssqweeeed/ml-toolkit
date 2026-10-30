"""Held-out permutation semantics, grouped columns and artifact behavior."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from ml_toolkit.config import FigureConfig
from ml_toolkit.metrics import MetricScores
from ml_toolkit.model_evaluation import PermutationImportance
from tests.support.training import MeanModel, make_data, mae


class SignalModel(MeanModel):
    def predict_pool(self, data):
        return data.X["x"].to_numpy() + 1


def inputs():
    data = make_data()
    data = replace(data, weight=pl.Series(np.arange(1.0, 25.0)))
    return data, pl.Series(["train", "val"] * 12), SignalModel({}).fit(data)


def test_selected_score_deterioration_weights_and_original_rows():
    data, split, model = inputs()
    original = data.X.clone()
    seen = []

    def metric(y, prediction, weight):
        seen.append((y.copy(), prediction.copy(), weight.copy()))
        return MetricScores(
            {
                "mae": float(np.average(abs(y - prediction), weights=weight)),
                "other": 9.0,
            }
        )

    report = PermutationImportance(score="mae", direction="minimize", n_repeats=3)
    table = report.calculate(model=model, metric=metric, data=data, split=split)
    assert table.height == 6
    assert set(table["split"]) == {"val"}
    assert set(table["metric"]) == {"mae"}
    assert len(seen) == 7
    assert table["baseline"].to_list() == [0.0] * 6
    assert table.filter(pl.col("group_id") == "g000")["importance"].min() > 0
    assert (
        table.filter(pl.col("group_id") == "g001")["importance"].to_list() == [0.0] * 3
    )
    for target, prediction, weight in seen:
        np.testing.assert_array_equal(target, data.y.to_numpy()[1::2])
        np.testing.assert_array_equal(weight, data.weight.to_numpy()[1::2])
    np.testing.assert_array_equal(
        table["value"].to_numpy(), table["importance"].to_numpy()
    )
    assert data.X.equals(original)
    assert report.calculate(model=model, metric=metric, data=data, split=split).equals(
        table
    )


def test_groups_use_common_permutation_and_exclude_grouped_default_features():
    data, split, _ = inputs()
    data = replace(data, X=data.X.with_columns(pl.col("x").alias("x_copy")))

    class GroupModel(SignalModel):
        def predict_pool(self, data):
            np.testing.assert_array_equal(data.X["x"], data.X["x_copy"])
            return super().predict_pool(data)

    report = PermutationImportance(
        score="mae", direction="minimize", groups=(("x", "x_copy"),), n_repeats=2
    )
    table = report.calculate(
        model=GroupModel({}).fit(data), metric=mae, data=data, split=split
    )
    assert table["features"].to_list() == [
        ["unused"],
        ["unused"],
        ["x", "x_copy"],
        ["x", "x_copy"],
    ]


def test_maximization_and_fixed_sampling_preserve_comparison_rows():
    data, split, model = inputs()

    def metric(y, prediction, weight):
        return MetricScores(
            {"negative_mae": -float(np.average(abs(y - prediction), weights=weight))}
        )

    report = PermutationImportance(
        score="negative_mae",
        direction="maximize",
        features=("x",),
        max_samples=5,
        n_repeats=3,
    )
    table = report.calculate(model=model, metric=metric, data=data, split=split)
    assert table["n_rows"].to_list() == [5] * 3
    assert table["sum_weight"].n_unique() == 1
    np.testing.assert_array_equal(
        table["importance"].to_numpy(), -table["value"].to_numpy()
    )
    assert report.calculate(model=model, metric=metric, data=data, split=split).equals(
        table
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"score": ""},
        {"direction": "up"},
        {"n_repeats": 0},
        {"max_samples": 0},
        {"seed": -1},
        {"groups": ((),)},
        {"features": ("x", "x")},
        {"features": ("x",), "groups": (("x",),)},
        {"features": ()},
    ],
)
def test_bad_settings(settings):
    with pytest.raises(ValueError):
        PermutationImportance(**{"score": "mae", "direction": "minimize", **settings})


def test_missing_features_and_score():
    data, split, model = inputs()
    with pytest.raises(ValueError, match="missing from X"):
        PermutationImportance(
            score="mae", direction="minimize", features=("missing",)
        ).calculate(model=model, metric=mae, data=data, split=split)
    with pytest.raises(ValueError, match="did not return score"):
        PermutationImportance(score="missing", direction="minimize").calculate(
            model=model, metric=mae, data=data, split=split
        )


def test_train_only_skips_and_nonfinite_scores_are_preserved_as_unavailable(tmp_path):
    data, split, model = inputs()
    report = PermutationImportance(score="mae", direction="minimize", n_repeats=2)
    train = pl.Series(["train"] * len(data))
    assert report.calculate(model=model, metric=mae, data=data, split=train).is_empty()
    assert (
        report.write_artifacts(
            model=model,
            metric=mae,
            data=data,
            split=train,
            artifact_dir=tmp_path,
            figure=FigureConfig(),
        )
        == []
    )
    undefined = lambda y, p, w: MetricScores({"mae": float("nan")})
    table = report.calculate(model=model, metric=undefined, data=data, split=split)
    assert table["importance"].null_count() == table.height
    assert report.evaluate(model=model, metric=mae, data=data, split=split) == {}
    paths = report.write_artifacts(
        model=model,
        metric=undefined,
        data=data,
        split=split,
        artifact_dir=tmp_path,
        figure=FigureConfig(format="svg"),
    )
    assert [path.name for path in paths] == ["importance.parquet", "importance.svg"]
    assert pl.read_parquet(paths[0]).equals(table)
    assert all(path.is_file() for path in paths)
    assert not plt.get_fignums()


def test_invalid_prediction_and_metric_exceptions_propagate():
    data, split, model = inputs()
    report = PermutationImportance(score="mae", direction="minimize")

    class Invalid(SignalModel):
        def predict_pool(self, data):
            return np.zeros((len(data), 2))

    with pytest.raises(ValueError, match="one finite value"):
        report.calculate(
            model=Invalid({}).fit(data), metric=mae, data=data, split=split
        )
    error = RuntimeError("metric error")

    def metric(*args):
        raise error

    with pytest.raises(RuntimeError) as caught:
        report.calculate(model=model, metric=metric, data=data, split=split)
    assert caught.value is error


def test_sampling_ignores_zero_weight_rows():
    data, split, model = inputs()
    data = replace(data, weight=pl.Series([0.0] * 23 + [1.0]))
    report = PermutationImportance(
        score="mae", direction="minimize", max_samples=1, seed=1
    )
    table = report.calculate(model=model, metric=mae, data=data, split=split)
    assert table["n_rows"].unique().to_list() == [1]
    assert table["sum_weight"].unique().to_list() == [1.0]
    assert table["importance"].to_list() == [0.0] * table.height


def test_nonfinite_importance_from_finite_scores_becomes_null():
    data, split, model = inputs()
    calls = []

    def metric(*args):
        calls.append(1)
        return MetricScores({"mae": -1e308 if len(calls) == 1 else 1e308})

    table = PermutationImportance(
        score="mae", direction="minimize", n_repeats=1
    ).calculate(model=model, metric=metric, data=data, split=split)
    assert table["importance"].null_count() == table.height
