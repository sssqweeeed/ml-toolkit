"""Execute the public configuration recipes with their documented data setup."""

import re
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest

from examples.render_examples import render

pytestmark = pytest.mark.documentation


@pytest.mark.parametrize("recipe_style", ["base", "full"])
@pytest.mark.parametrize(
    "task",
    [
        "Binary classification",
        "Regression",
        "Frequency",
        "Severity",
        "Positive–unlabeled classification",
    ],
)
def test_documented_task_recipes_train_and_evaluate(
    task, recipe_style, tmp_path, monkeypatch
):
    docs = Path(__file__).resolve().parents[2] / "docs"
    document = (docs / "configuration.md").read_text()
    sections = dict(
        re.findall(
            r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", document, re.MULTILINE | re.DOTALL
        )
    )
    namespace = {}
    data_block = re.findall(
        r"```python\n(.*?)\n```", (docs / "api/data.md").read_text(), re.DOTALL
    )[0]
    exec(data_block, namespace)
    if task in {"Binary classification", "Positive–unlabeled classification"}:
        data = namespace["data"]
        namespace["data"] = data.with_target((data.y > 0).cast(pl.Float64))
    task_blocks = re.findall(r"```python\n(.*?)\n```", sections[task], re.DOTALL)
    if recipe_style == "base":
        for block in re.findall(
            r"```python\n(.*?)\n```", sections["Imports and shared behavior"], re.DOTALL
        ):
            exec(block, namespace)
        exec(task_blocks[0], namespace)
    else:
        pytest.importorskip("shap", reason="Full recipes enable SHAP diagnostics")
        # The expanded recipe must work without any shared configuration setup.
        exec(task_blocks[1], namespace)

    monkeypatch.chdir(tmp_path)
    set_uri = mlflow.set_tracking_uri
    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        lambda uri: set_uri(f"sqlite:///{tmp_path}/mlflow.db"),
    )
    run_block = re.findall(
        r"```python\n(.*?)\n```", sections["Run the chosen recipe"], re.DOTALL
    )[0]
    exec(run_block, namespace)

    result = namespace["result"]
    assert len(result.validation.splits) == 3
    assert result.validation.mean_metrics
    assert result.test_metrics
    assert all(np.isfinite(value) for value in result.test_metrics.values())
    selected_data = namespace["data"].select_features(result.features)
    predictions = result.model.predict_pool(selected_data)
    assert predictions.shape == (24,)
    assert np.isfinite(predictions).all()
    if recipe_style == "full":
        for split in result.validation.splits:
            assert split.feature_selection is not None
            assert len(split.feature_selection.steps) == 2
            assert split.feature_selection.steps[0].features == ("age", "region")
            assert len(split.feature_selection.steps[-1].features) == 1
            assert split.tuning is not None
            assert np.isfinite(split.tuning.best_score)
            assert "learning_rate" in split.tuning.best_params
        assert result.features == result.validation.last_split.features
        client = mlflow.MlflowClient()
        experiment = mlflow.get_experiment_by_name("ml-toolkit-task-recipes")
        runs = client.search_runs([experiment.experiment_id])
        final_run = next(
            run for run in runs if run.data.tags["mlflow.runName"] == "final_fit"
        )
        figures = client.list_artifacts(final_run.info.run_id, "figures/shap/test")
        assert {item.path for item in figures} == {
            "figures/shap/test/importance.png",
            "figures/shap/test/beeswarm.png",
        }
        models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{final_run.info.run_id}'",
            output_format="list",
        )
        assert len(models) == 1
        reloaded = mlflow.pyfunc.load_model(models[0].model_uri)
        np.testing.assert_allclose(
            reloaded.predict(selected_data.to_model_input()), predictions
        )
    else:
        assert result.feature_selection is None
        assert result.tuning is None


def test_complete_frequency_example_output_and_figure():
    from examples.gallery_data import bike_cache_available

    if not bike_cache_available():
        pytest.skip("The complete frequency example needs cached UCI Bike Sharing data")
    pytest.importorskip("shap", reason="Full recipes enable SHAP diagnostics")
    render(check=True, pages=("docs/configuration.md",))
