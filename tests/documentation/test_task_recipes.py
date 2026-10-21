"""Execute the public configuration recipes with their documented data setup."""

import re
from pathlib import Path

import mlflow
import numpy as np
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
    ],
)
def test_documented_task_recipes_train_and_evaluate(
    task, recipe_style, tmp_path, monkeypatch
):
    from examples.gallery_data import bike_cache_available
    from examples.retail_data import retail_cache_available

    available = retail_cache_available() if task == "Severity" else bike_cache_available()
    if not available:
        pytest.skip("Prepare real source data with python -m examples.prepare_data")
    docs = Path(__file__).resolve().parents[2] / "docs"
    document = (docs / "configuration.md").read_text()
    sections = dict(
        re.findall(
            r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", document, re.MULTILINE | re.DOTALL
        )
    )
    namespace = {}
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
    assert len(result.validation.splits) == (1 if task == "Severity" else 3)
    assert result.validation.mean_metrics
    assert result.test_metrics
    assert all(np.isfinite(value) for value in result.test_metrics.values())
    selected_data = namespace["data"].select_features(result.features)
    predictions = result.model.predict_pool(selected_data)
    assert predictions.shape == (len(namespace["data"]),)
    assert len(namespace["data"]) > 1000
    assert namespace["data"].time.min().year in {2010, 2011}
    if task == "Binary classification":
        assert set(namespace["data"].y.unique()) == {0.0, 1.0}
    if task == "Severity":
        assert (namespace["data"].y > 0).all()
        assert namespace["config"].test_size.isoformat() == "2011-10-01"
    assert np.isfinite(predictions).all()
    if recipe_style == "full":
        for split in result.validation.splits:
            assert split.feature_selection is not None
            assert len(split.feature_selection.steps) == 2
            assert split.feature_selection.steps[0].features == tuple(namespace["data"].X.columns)
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

    from examples.retail_data import retail_cache_available

    if not bike_cache_available() or not retail_cache_available():
        pytest.skip("Configuration recipes require cached Bike Sharing and Online Retail data")
    pytest.importorskip("shap", reason="Full recipes enable SHAP diagnostics")
    render(check=True, pages=("docs/configuration.md",))


@pytest.mark.parametrize("recipe_style", ["base", "full"])
def test_pu_recipe_requires_real_pu_data_and_does_not_train_on_supervised_targets(recipe_style):
    docs = Path(__file__).resolve().parents[2] / "docs"
    document = (docs / "configuration.md").read_text()
    sections = dict(re.findall(r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", document, re.MULTILINE | re.DOTALL))
    namespace = {}
    if recipe_style == "base":
        for block in re.findall(r"```python\n(.*?)\n```", sections["Imports and shared behavior"], re.DOTALL):
            exec(block, namespace)
    blocks = re.findall(r"```python\n(.*?)\n```", sections["Positive–unlabeled classification"], re.DOTALL)
    exec(blocks[0 if recipe_style == "base" else 1], namespace)
    assert "data" not in namespace
    assert namespace["params"]["pu_prior"] == 0.2
    assert namespace["config"].reporting.diagnostics.plots.common.observed_fitted is False
    assert "configuration assumption" in sections["Positive–unlabeled classification"]
