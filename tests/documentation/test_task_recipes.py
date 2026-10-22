"""Execute the public configuration recipes with their documented data setup."""

from pathlib import Path

import mlflow
import numpy as np
import pytest

from examples.render_examples import execute, recipes, render

pytestmark = pytest.mark.documentation


@pytest.mark.parametrize(
    "recipe_id,full",
    [
        ("configuration-classification", False),
        ("configuration-regression", False),
        ("configuration-frequency", False),
        ("configuration-severity", False),
        ("configuration-frequency-full", True),
    ],
)
def test_documented_task_recipes_train_and_evaluate(
    recipe_id, full, tmp_path, monkeypatch
):
    from examples.gallery_data import bike_cache_available
    from examples.retail_data import retail_cache_available

    severity = recipe_id == "configuration-severity"
    available = retail_cache_available() if severity else bike_cache_available()
    if not available:
        pytest.skip("Prepare real source data with python -m examples.prepare_data")
    if full:
        pytest.importorskip("shap", reason="The complete recipe enables SHAP")
    page = Path(__file__).resolve().parents[2] / "docs/configuration.md"
    snippets = {item.name: item for item in recipes(page)}
    namespace = {}
    monkeypatch.chdir(tmp_path)
    set_uri = mlflow.set_tracking_uri
    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        lambda uri: set_uri(f"sqlite:///{tmp_path}/mlflow.db"),
    )
    if not full:
        execute(snippets["configuration-shared"], namespace)
    execute(snippets[recipe_id], namespace)
    execute(snippets["configuration-run"], namespace)

    result = namespace["result"]
    assert len(result.validation.splits) == (1 if severity else 3)
    assert result.validation.mean_metrics
    assert result.test_metrics
    assert all(np.isfinite(value) for value in result.test_metrics.values())
    selected_data = namespace["data"].select_features(result.features)
    predictions = result.model.predict_pool(selected_data)
    assert predictions.shape == (len(namespace["data"]),)
    assert len(namespace["data"]) > 1000
    assert namespace["data"].time.min().year in {2010, 2011}
    if recipe_id == "configuration-classification":
        assert set(namespace["data"].y.unique()) == {0.0, 1.0}
    if severity:
        assert (namespace["data"].y > 0).all()
        assert namespace["splitter"].test_size.isoformat() == "2011-10-01"
    assert np.isfinite(predictions).all()
    if full:
        for split in result.validation.splits:
            assert split.feature_selection is not None
            assert len(split.feature_selection.steps) == 2
            assert split.feature_selection.steps[0].features == tuple(
                namespace["data"].X.columns
            )
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
        figures = client.list_artifacts(final_run.info.run_id, "shap/test")
        assert {item.path for item in figures} == {
            "shap/test/importance.png",
            "shap/test/beeswarm.png",
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
        pytest.skip(
            "Configuration recipes require cached Bike Sharing and Online Retail data"
        )
    pytest.importorskip("shap", reason="Full recipes enable SHAP diagnostics")
    render(check=True, pages=("docs/configuration.md",))


def test_pu_recipe_requires_project_data_and_an_explicit_prior():
    page = Path(__file__).resolve().parents[2] / "docs/configuration.md"
    snippets = {item.name: item for item in recipes(page)}
    namespace = {}
    execute(snippets["configuration-shared"], namespace)
    execute(snippets["configuration-pu"], namespace)
    assert "data" not in namespace
    with pytest.raises(TypeError, match="data"):
        namespace["train_pu"]()
    with pytest.raises(TypeError, match="prior"):
        namespace["train_pu"](object())
