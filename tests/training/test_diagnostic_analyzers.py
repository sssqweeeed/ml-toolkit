"""New model and prediction reports compose in the normal training workflow."""

from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from ml_toolkit.config import TrackingConfig, TrainingConfig
from ml_toolkit.evaluation import MetricTrend, Residuals
from ml_toolkit.model_evaluation import PermutationImportance, RobustnessCurve
from ml_toolkit.training import TimeSeriesCV, train_model
from tests.support.training import (
    MeanModel,
    list_all_artifacts,
    mae,
    make_identified_data,
    setup_mlflow,
)


def test_four_reports_compose_without_shrink_capability_or_model_changes(tmp_path):
    setup_mlflow(tmp_path)
    data = make_identified_data()
    original = data.X.clone()
    result = train_model(
        splitter=TimeSeriesCV(data, n_splits=2, test_size=4),
        params={},
        model_factory=MeanModel,
        metric=mae,
        run_name="four-reports",
        config=TrainingConfig(
            fit_on_test=True,
            tracking=TrackingConfig(log_test_model=False, log_final_model=False),
        ),
        model_evaluators=[
            PermutationImportance(score="mae", direction="minimize", n_repeats=2),
            RobustnessCurve(features=("x",), levels=(0.0, 0.5), n_repeats=2),
        ],
        prediction_evaluators=[
            MetricTrend(metric=mae, period="quarter", window=2),
            Residuals(n_bins=3, max_rows=2),
        ],
    )
    assert data.X.equals(original)
    assert result.test_metrics == {"mae": 12.0}
    assert result.model.params == {}
    np.testing.assert_array_equal(
        result.model.predict_pool(data), np.full(len(data), 10.5)
    )
    client = mlflow.MlflowClient()
    fits = 0
    for run in mlflow.search_runs(output_format="list"):
        name = run.data.tags["mlflow.runName"]
        if name not in (
            "validation/split_1",
            "validation/split_2",
            "test",
            "final_fit",
        ):
            continue
        fits += 1
        files = list_all_artifacts(client, run.info.run_id)
        assert "metric_trend/metrics.parquet" in files
        assert "residuals/rows.parquet" in files
        assert "residuals/residuals.png" in files
        if name == "final_fit":
            assert not any(
                path.startswith(("permutation_importance/", "robustness/"))
                for path in files
            )
        else:
            assert "permutation_importance/importance.parquet" in files
            assert "robustness/metrics.parquet" in files
        if name == "test":
            path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path="residuals/rows.parquet"
            )
            rows = pl.read_parquet(Path(path))
            assert rows.filter(pl.col("split") == "test")["object_id"].to_list() == [
                20,
                21,
                22,
                23,
            ]
    assert fits == 4
