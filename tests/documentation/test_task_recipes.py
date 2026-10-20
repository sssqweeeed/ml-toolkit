"""Execute the public configuration recipes with their documented data setup."""

import re
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest

from examples.render_examples import recipes

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
    predictions = result.model.predict_pool(namespace["data"])
    assert predictions.shape == (24,)
    assert np.isfinite(predictions).all()


def test_full_configurations_match_base_recipes():
    path = Path(__file__).resolve().parents[2] / "docs/configuration.md"
    snippets = {recipe.name: recipe.code for recipe in recipes(path)}
    for task in ("classification", "regression", "frequency", "severity", "pu"):
        compact = {}
        exec(snippets["configuration-shared"], compact)
        exec(snippets[f"configuration-{task}"], compact)
        full = {}
        exec(snippets[f"configuration-{task}-full"], full)
        assert full["config"] == compact["config"]
        assert full["params"] == compact["params"]
        assert full["model_factory"] is compact["model_factory"]
