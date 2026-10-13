import json
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from test_training import list_all_artifacts, make_data, setup_mlflow

from ml_toolkit.config import (
    ClassificationDiagnosticsConfig,
    ClassificationPlotsConfig,
    CommonPlotsConfig,
    FigureConfig,
    PUDiagnosticsConfig,
    PUPlotsConfig,
    RegressionDiagnosticsConfig,
    RegressionPlotsConfig,
    ShapConfig,
)
from ml_toolkit.models import CatBoostRegressionModel
from ml_toolkit.training import DatasetSplit, reporting


@pytest.mark.parametrize("split", [DatasetSplit.VAL, DatasetSplit.TEST])
@pytest.mark.parametrize(
    "task, format", [("regression", "png"), ("classification", "svg"), ("pu", "pdf")]
)
def test_diagnostic_layout_preserves_all_images_and_tables(
    tmp_path, monkeypatch, split, task, format
):
    pytest.importorskip("shap")
    setup_mlflow(tmp_path)
    data = make_data().with_target(pl.Series("target", [0.0, 1.0] * 12))
    train, current = data.take(range(12)), data.take(range(12, 24))
    model = CatBoostRegressionModel(
        {"iterations": 3, "depth": 2, "verbose": False, "allow_writing_files": False}
    ).fit(train)
    common = CommonPlotsConfig(
        n_bins=3,
        figure=FigureConfig(format=format, dpi=72),
        shap=ShapConfig(max_samples=12, dependence_features=("x", "unused")),
    )
    if task == "classification":
        options = ClassificationDiagnosticsConfig(
            plots=ClassificationPlotsConfig(common=common)
        )
        performance = {"gains", "lift", "precision_recall", "f1_threshold", "calibration"}
    elif task == "pu":
        options = PUDiagnosticsConfig(plots=PUPlotsConfig(common=common))
        performance = {"score_distribution", "positive_capture"}
    else:
        options = RegressionDiagnosticsConfig(
            plots=RegressionPlotsConfig(common=common, gains=True, lift=True)
        )
        performance = {"gains", "lift"}

    tables = {}
    original_log_table = reporting.log_polars_table

    def log_table(table, artifact_file):
        tables[artifact_file] = json.loads(json.dumps(table.to_dicts(), default=str))
        original_log_table(table, artifact_file)

    monkeypatch.setattr(reporting, "log_polars_table", log_table)
    with mlflow.start_run() as run:
        reporting.report_diagnostics(
            model,
            train,
            current,
            model.predict_pool(train),
            model.predict_pool(current),
            options,
            "layout",
            split=split,
        )

    suffix = f".{format}"
    expected_images = {
        f"figures/psi/{split.value}/scores{suffix}",
        f"figures/shap/{split.value}/importance{suffix}",
        f"figures/shap/{split.value}/beeswarm{suffix}",
    }
    collections = ("psi", "observed_fitted", "feature_distributions", "shap")
    directories = ("monthly_by_feature", "by_feature", "monthly_by_feature", "dependence")
    for category, directory in zip(collections, directories, strict=True):
        expected_images.update(
            f"figures/{category}/{split.value}/{directory}/{feature}{suffix}"
            for feature in ("000_x", "001_unused")
        )
    expected_images.update(
        f"figures/performance/{split.value}/{name}{suffix}" for name in performance
    )
    if task != "pu":
        expected_images.add(f"figures/observed_fitted/{split.value}/monthly{suffix}")
    expected_tables = {
        f"tables/psi/{split.value}/{name}.json"
        for name in ("scores", "level_distribution", "monthly_distribution")
    }
    client = mlflow.MlflowClient()
    assert set(list_all_artifacts(client, run.info.run_id)) == expected_images | expected_tables
    download_dir = tmp_path / "download"
    download_dir.mkdir()
    downloaded = Path(client.download_artifacts(run.info.run_id, "", str(download_dir)))
    images = {path: (downloaded / path).read_bytes() for path in expected_images}
    for content in images.values():
        if format == "png":
            assert content.startswith(b"\x89PNG\r\n\x1a\n")
        elif format == "pdf":
            assert content.startswith(b"%PDF-")
        else:
            assert b"<svg" in content
    # The same feature filename must retain a distinct image in each category.
    assert len({
        images[f"figures/{category}/{split.value}/{directory}/000_x{suffix}"]
        for category, directory in zip(collections, directories, strict=True)
    }) == len(collections)
    assert set(tables) == expected_tables
    for path in expected_tables:
        assert tables[path]
        assert json.loads((downloaded / path).read_text()) == {"rows": tables[path]}


def test_feature_filenames_preserve_order_and_disambiguate_sanitized_names(
    tmp_path, monkeypatch
):
    setup_mlflow(tmp_path)
    source = make_data()
    names = ("x / value", "x ? value", "." * 4, "x" * 80)
    data = replace(source, X=pl.DataFrame({name: source.X["x"] for name in names}))
    images = []

    def upload(path, *, artifact_path):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        images.append(f"{artifact_path}/{path.name}")

    monkeypatch.setattr(reporting.tracking, "log_artifact", upload)
    reporting.report_diagnostics(
        CatBoostRegressionModel({}),
        data,
        data,
        np.ones(len(data)),
        np.ones(len(data)),
        RegressionDiagnosticsConfig(
            plots=RegressionPlotsConfig(
                common=CommonPlotsConfig(psi=False, numerical_distributions=None),
                monthly_observed_fitted=False,
            )
        ),
        "filenames",
    )
    assert images == [
        f"figures/observed_fitted/val/by_feature/{filename}.png"
        for filename in ("000_x_value", "001_x_value", "002_feature", "003_" + "x" * 60)
    ]
