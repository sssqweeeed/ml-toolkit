from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from test_training import MeanModel, make_data

from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    FeatureLevelSpec,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
)
from ml_toolkit.feature_levels import LevelTransformer
from ml_toolkit.plots.regression import observed_fitted_by_level
from ml_toolkit.plots.utils import PlotError
from ml_toolkit.training import DatasetSplit
from ml_toolkit.training import reporting


def diagnostics_for(task, common):
    if task == "classification":
        return ClassificationDiagnosticsConfig(
            plots=ClassificationPlotsConfig(
                common=common,
                precision_recall=False,
                f1_threshold=False,
                calibration=False,
                gains=False,
                lift=False,
                monthly_observed_fitted=False,
            )
        )
    if task == "pu":
        return PUDiagnosticsConfig(
            plots=PUPlotsConfig(
                common=common, score_distribution=False, positive_capture=False
            )
        )
    return RegressionDiagnosticsConfig(
        plots=RegressionPlotsConfig(common=common, monthly_observed_fitted=False)
    )


@pytest.mark.parametrize("task", ["regression", "classification", "pu"])
@pytest.mark.parametrize(
    "psi, observed_fitted", [(True, True), (True, False), (False, True), (False, False)]
)
def test_shared_diagnostics_switches_and_single_reference_fit(
    monkeypatch, task, psi, observed_fitted
):
    data = make_data()
    train, current = data.take(range(12)), data.take(range(12, 24))
    artifacts, tables, fits, transforms, comparisons, summaries = [], {}, [], [], [], []
    original_fit = LevelTransformer.fit
    original_transform = LevelTransformer.transform
    original_compare = reporting.compare_psi
    original_plot = reporting.regression.observed_fitted_by_level

    def fit(self, frame):
        fits.append(frame)
        return original_fit(self, frame)

    def transform(self, frame):
        result = original_transform(self, frame)
        transforms.append(result)
        return result

    def compare(reference, current, config, **kwargs):
        comparisons.append((reference, current, config))
        return original_compare(reference, current, config, **kwargs)

    def plot(*args, **kwargs):
        summary = original_plot(*args, **kwargs)
        summaries.append((args, kwargs, summary))
        return summary

    def log_artifact(path, *, artifact_path):
        assert path.is_file() and path.stat().st_size > 0
        artifacts.append(f"{artifact_path}/{path.name}")

    monkeypatch.setattr(LevelTransformer, "fit", fit)
    monkeypatch.setattr(LevelTransformer, "transform", transform)
    monkeypatch.setattr(reporting, "compare_psi", compare)
    monkeypatch.setattr(reporting.regression, "observed_fitted_by_level", plot)
    monkeypatch.setattr(reporting.tracking, "log_artifact", log_artifact)
    monkeypatch.setattr(
        reporting, "log_polars_table", lambda table, path: tables.update({path: table})
    )
    common = CommonPlotsConfig(
        features=(
            FeatureLevelSpec("unused", "categorical"),
            FeatureLevelSpec("x", "numerical"),
        ),
        psi=psi,
        observed_fitted=observed_fitted,
        n_bins=3,
        numerical_distributions=None,
    )
    reporting.report_diagnostics(
        MeanModel({}),
        train,
        current,
        np.full(len(train), 2.0),
        np.full(len(current), 3.0),
        diagnostics_for(task, common),
        "test",
        split=DatasetSplit.TEST,
    )
    if psi or observed_fitted:
        assert len(fits) == 1 and fits[0].equals(train.X)
        assert len(transforms) == 2
    else:
        assert not fits and not transforms
    assert bool(tables) == psi
    if psi:
        assert tables["test/psi/scores.json"]["feature"].to_list() == ["unused", "x"]
        assert comparisons[0][0] is transforms[0]
        assert comparisons[0][1] is transforms[1]
        assert "test/figures/psi_000_unused.png" in artifacts
        assert "test/figures/psi_001_x.png" in artifacts
    else:
        assert not comparisons
    assert len(summaries) == (2 if observed_fitted else 0)
    if observed_fitted:
        assert "test/figures/observed_fitted_000_unused.png" in artifacts
        assert "test/figures/observed_fitted_001_x.png" in artifacts
        for args, kwargs, summary in summaries:
            name = kwargs["feature_name"]
            assert args[0].equals(transforms[0][name])
            assert args[3].equals(transforms[1][name])
            assert kwargs["current_label"] == "Test"
            assert (
                summary.loc[summary["sample"] == "reference", "volume"].sum()
                == train.weight.sum()
            )
            assert (
                summary.loc[summary["sample"] == "current", "volume"].sum()
                == current.weight.sum()
            )
        # Validation values beyond the train range retain the train's last bin.
        assert transforms[1]["x"].n_unique() == 1
    if not psi and not observed_fitted:
        assert artifacts == []


@pytest.mark.parametrize("disabled", ["empty", "plots_none"])
def test_disabled_feature_diagnostics_do_not_fit(monkeypatch, disabled):
    data = make_data()

    def unexpected(*args, **kwargs):
        pytest.fail(
            "Disabled feature diagnostics must not prepare levels or log artifacts"
        )

    monkeypatch.setattr(LevelTransformer, "fit", unexpected)
    monkeypatch.setattr(reporting.tracking, "log_artifact", unexpected)
    monkeypatch.setattr(reporting, "log_polars_table", unexpected)
    options = diagnostics_for(
        "regression", CommonPlotsConfig(features=(), numerical_distributions=None)
    )
    if disabled == "plots_none":
        options = RegressionDiagnosticsConfig(plots=None)
    reporting.report_diagnostics(
        MeanModel({}),
        data,
        data,
        np.ones(len(data)),
        np.ones(len(data)),
        options,
        "disabled",
    )


def test_default_feature_inference_and_missing_feature(monkeypatch):
    data = replace(make_data(), cat_features=("unused",))
    captured = []
    original_fit = LevelTransformer.fit

    def fit(self, frame):
        captured.extend(self.features)
        return original_fit(self, frame)

    monkeypatch.setattr(LevelTransformer, "fit", fit)
    monkeypatch.setattr(
        reporting.tracking, "log_artifact", lambda *args, **kwargs: None
    )
    common = CommonPlotsConfig(psi=False, numerical_distributions=None)
    reporting.report_diagnostics(
        MeanModel({}),
        data,
        data,
        np.ones(len(data)),
        np.ones(len(data)),
        diagnostics_for("regression", common),
        "inference",
    )
    assert [(f.name, f.ftype) for f in captured] == [
        ("x", "numerical"),
        ("unused", "categorical"),
    ]
    missing = replace(common, features=(FeatureLevelSpec("missing", "numerical"),))
    with pytest.raises(ValueError, match="missing"):
        reporting.report_diagnostics(
            MeanModel({}),
            data,
            data,
            np.ones(len(data)),
            np.ones(len(data)),
            diagnostics_for("regression", missing),
            "missing",
        )


def test_weighted_comparison_keeps_level_order_and_gaps(tmp_path):
    order = ("[0, 2)", "[2, 10)", "[10, inf)", "__NULL__", "__ERROR__", "__UNSEEN__")
    summary = observed_fitted_by_level(
        pl.Series([order[2], order[0], order[0], order[3], order[4]]),
        [10, 1, 5, 7, 99],
        [9, 2, 6, 8, 99],
        pl.Series([order[1], order[2], order[5]]),
        [3, 12, 4],
        [4, 11, 5],
        tmp_path / "comparison.png",
        level_order=order,
        reference_weight=[2, 1, 3, 1, 0],
        current_weight=[1, 2, 1],
    )
    reference = summary.loc[summary["sample"] == "reference"].set_index("level")
    current = summary.loc[summary["sample"] == "current"].set_index("level")
    assert list(reference.index) == [order[0], order[1], order[2], order[3], order[5]]
    assert reference.loc[order[0], "observed"] == 4.0
    assert reference.loc[order[0], "fitted"] == 5.0
    assert reference.loc[order[0], "volume"] == 4.0
    assert np.isnan(reference.loc[order[1], "observed"])
    assert reference.loc[order[1], "volume"] == 0
    assert np.isnan(current.loc[order[0], "fitted"])
    assert (tmp_path / "comparison.png").stat().st_size > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_observed": [1]},
        {"current_fitted": [1, np.inf]},
        {"reference_weight": [-1, 2]},
        {"current_weight": [0, 0]},
        {"level_order": ("a", "a")},
        {"current_levels": pl.Series(["a", "unknown"])},
        {"reference_levels": pl.Series(["a", None])},
    ],
)
def test_observed_fitted_rejects_invalid_inputs(tmp_path, kwargs):
    inputs = dict(
        reference_levels=pl.Series(["a", "b"]),
        reference_observed=[1, 2],
        reference_fitted=[1, 2],
        current_levels=pl.Series(["a", "b"]),
        current_observed=[1, 2],
        current_fitted=[1, 2],
        level_order=("a", "b"),
        save_fig=tmp_path / "invalid.png",
    )
    inputs.update(kwargs)
    with pytest.raises(PlotError):
        observed_fitted_by_level(**inputs)


def test_shared_config_rejects_duplicate_features():
    with pytest.raises(ValueError, match="unique"):
        CommonPlotsConfig(
            features=(
                FeatureLevelSpec("x", "numerical"),
                FeatureLevelSpec("x", "categorical"),
            )
        )


def test_special_levels_and_constant_features_are_shared_with_psi(tmp_path):
    from ml_toolkit.psi import compare_psi

    reference = pl.DataFrame(
        {
            "value": [0, 0, 10, 10, None, -1, -999],
            "group": ["a", "b", "a", "b", None, "missing", "error"],
            "constant": [5] * 7,
        }
    )
    current = pl.DataFrame(
        {
            "value": [-2, 0, 10, 21, None, -1, -999],
            "group": ["a", "new", "b", "new", None, "missing", "error"],
            "constant": [5] * 7,
        }
    )
    transformer = LevelTransformer(
        (
            FeatureLevelSpec(
                "value",
                "numerical",
                lower_bound=0,
                upper_bound=20,
                null_level=-1,
                error_level=-999,
            ),
            FeatureLevelSpec(
                "group", "categorical", null_level="missing", error_level="error"
            ),
            FeatureLevelSpec("constant", "numerical"),
        ),
        n_bins=10,
    ).fit(reference)
    ref_levels, cur_levels = transformer.transform(reference), transformer.transform(
        current
    )
    assert transformer.config is not None
    assert len(transformer.config.features[0].boundaries) < 9
    assert transformer.config.features[2].boundaries == ()
    assert cur_levels["value"].cast(pl.String).to_list()[3:] == [
        "__ERROR__",
        "__NULL__",
        "__NULL__",
        "__ERROR__",
    ]
    assert cur_levels["group"].cast(pl.String).to_list() == [
        "a",
        "__UNSEEN__",
        "b",
        "__UNSEEN__",
        "__NULL__",
        "__NULL__",
        "__ERROR__",
    ]
    result = compare_psi(ref_levels, cur_levels, transformer.config)
    for feature in transformer.config.features:
        table = observed_fitted_by_level(
            ref_levels[feature.name],
            np.arange(7),
            np.arange(7),
            cur_levels[feature.name],
            np.arange(7),
            np.arange(7),
            tmp_path / f"{feature.name}.png",
            level_order=feature.levels,
        )
        distribution = result.level_distribution.filter(
            pl.col("feature") == feature.name
        )
        for row in table.itertuples():
            count = distribution.filter(
                (pl.col("sample") == row.sample) & (pl.col("level") == row.level)
            )["rows"].item()
            assert row.volume == count
