from dataclasses import replace

import numpy as np
import pytest

from tests.support.training import MeanModel, make_data
from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    FeatureLevelConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
)
from ml_toolkit.feature_levels import LevelTransformer
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
                time_observed_fitted=False,
            )
        )
    if task == "pu":
        return PUDiagnosticsConfig(
            plots=PUPlotsConfig(
                common=common, score_distribution=False, positive_capture=False
            )
        )
    return RegressionDiagnosticsConfig(
        plots=RegressionPlotsConfig(common=common, time_observed_fitted=False)
    )


@pytest.mark.parametrize(
    "task,psi,observed_fitted",
    [
        ("regression", True, True),
        ("regression", True, False),
        ("regression", False, True),
        ("regression", False, False),
        ("classification", True, True),
        ("pu", True, True),
    ],
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
            FeatureLevelConfig("unused", "categorical"),
            FeatureLevelConfig("x", "numerical"),
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
        assert tables["tables/psi/scores.json"]["feature"].to_list() == [
            "unused",
            "x",
        ]
        assert comparisons[0][0] is transforms[0]
        assert comparisons[0][1] is transforms[1]
        assert "figures/psi/time_by_feature/000_unused.png" in artifacts
        assert "figures/psi/time_by_feature/001_x.png" in artifacts
    else:
        assert not comparisons
    assert len(summaries) == (2 if observed_fitted else 0)
    if observed_fitted:
        assert "figures/observed_fitted/by_feature/000_unused.png" in artifacts
        assert "figures/observed_fitted/by_feature/001_x.png" in artifacts
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
    missing = replace(common, features=(FeatureLevelConfig("missing", "numerical"),))
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


def test_shared_config_rejects_duplicate_features():
    with pytest.raises(ValueError, match="unique"):
        CommonPlotsConfig(
            features=(
                FeatureLevelConfig("x", "numerical"),
                FeatureLevelConfig("x", "categorical"),
            )
        )
