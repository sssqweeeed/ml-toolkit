"""Artifacts frozen by ml-toolkit 6.2.x still load and reproduce their outputs."""

import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml_toolkit.feature_levels import LevelTransformer
from ml_toolkit.psi import PSIBaseline

FROZEN = Path(__file__).parent / "legacy_6_2"
MODELS = [
    "catboost_regression",
    "catboost_log_link",
    "catboost_classification",
    "catboost_pu_upu",
    "catboost_pu_nnpu",
    "inflation_adjusted",
    "lightgbm",
]


@pytest.mark.parametrize("name", MODELS)
def test_frozen_model_predicts_stored_values(name, tmp_path):
    if name == "lightgbm":
        pytest.importorskip("lightgbm")
    # Copy so that loading can never write into the repository.
    shutil.copytree(FROZEN / name, tmp_path / name)
    model = mlflow.pyfunc.load_model(str(tmp_path / name / "model"))
    prediction = model.predict(pd.read_parquet(tmp_path / name / "input.parquet"))
    expected = pd.read_parquet(tmp_path / name / "expected.parquet")
    np.testing.assert_allclose(
        np.asarray(prediction, dtype=float).ravel(),
        expected.to_numpy(dtype=float).ravel(),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "name, loss", [("catboost_pu_upu", "uPU"), ("catboost_pu_nnpu", "nnPU")]
)
def test_frozen_pu_state_version_1_maps_pu_prior_to_class_prior(name, loss, tmp_path):
    shutil.copytree(FROZEN / name, tmp_path / name)
    state = json.loads((tmp_path / name / "model/artifacts/config.json").read_text())
    assert state["version"] == 1 and "pu_prior" in state
    loaded = mlflow.pyfunc.load_model(
        str(tmp_path / name / "model")
    ).unwrap_python_model()
    assert loaded.class_prior == 0.3
    assert loaded.pu_loss == loss
    # Training parameters are kept verbatim, so refitting fails loudly.
    assert "pu_prior" in loaded.params


def test_frozen_psi_baseline_compare_matches(tmp_path):
    shutil.copytree(FROZEN / "psi", tmp_path / "psi")
    root = tmp_path / "psi"
    baseline = PSIBaseline.load(root / "baseline")
    result = baseline.compare(pl.read_parquet(root / "current.parquet"))
    assert_frame_equal(result.psi, pl.read_parquet(root / "expected_psi.parquet"))
    assert_frame_equal(
        result.level_distribution,
        pl.read_parquet(root / "expected_level_distribution.parquet"),
    )


def test_frozen_level_transformer_transforms_identically(tmp_path):
    shutil.copytree(FROZEN / "levels", tmp_path / "levels")
    root = tmp_path / "levels"
    transformer = LevelTransformer.from_dict(
        json.loads((root / "transformer.json").read_text(encoding="utf-8"))
    )
    assert_frame_equal(
        transformer.transform(pl.read_parquet(root / "input.parquet")),
        pl.read_parquet(root / "expected.parquet"),
    )
