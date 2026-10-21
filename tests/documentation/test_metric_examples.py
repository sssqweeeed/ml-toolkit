"""Small calculation recipes teach observable behavior without gallery assets."""

from pathlib import Path

import numpy as np
import pytest

from examples.render_examples import RECIPE, Recipe, execute

pytestmark = pytest.mark.documentation
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def metric_examples():
    namespace = {}
    document = (ROOT / "docs/api/metrics.md").read_text()
    for match in RECIPE.finditer(document):
        actual = execute(Recipe(match["name"], match["code"]), namespace)
        expected = match["output"].split("```text\n", 1)[1].split("\n```", 1)[0]
        assert actual == expected, f"Stale metric output: {match['name']}"
    return namespace


def test_threshold_example_exposes_weighted_precision_recall_tradeoff(metric_examples):
    ns = metric_examples
    metric = ns["metrics_classification"]
    low = metric(ns["observed"], ns["probabilities"], threshold=0.3, sample_weight=ns["weight"])
    high = ns["scores"]
    assert low.precision == pytest.approx(3 / 5)
    assert low.recall == 1
    assert high.precision > low.precision
    assert high.recall == pytest.approx(2 / 3)
    assert high.recall < low.recall


def test_regression_examples_reveal_weighting_and_underprediction(metric_examples):
    ns = metric_examples
    frequency = ns["frequency"]
    errors = np.abs(ns["frequency_observed"] - ns["frequency_prediction"])
    assert frequency.mean_absolute_error == pytest.approx(1.3 / 3.5)
    assert frequency.mean_absolute_error != pytest.approx(errors.mean())
    assert frequency.prediction_over_observed == pytest.approx(4.3 / 5)
    assert frequency.mean_poisson_deviance > 0
    severity = ns["severity"]
    assert severity.mean_absolute_error == 40
    assert severity.prediction_over_observed == pytest.approx(1200 / 1400)
    assert severity.mean_tweedie_deviance > 0
    assert ns["bound_scores"].to_dict() == severity.to_dict()


def test_ranking_example_exposes_imperfect_order_and_exposure_capacity(metric_examples):
    ranking = metric_examples["ranking"]
    assert ranking.normalized_gini == pytest.approx(3 / 7)
    first, second = ranking.selected_points
    assert first.percent == 25
    assert first.lift < 1
    assert second.percent == 50
    assert second.gain == pytest.approx(10 / 17)
    assert second.lift > 1
    np.testing.assert_allclose(ranking.cumulative_exposure, [0, 1 / 6, 3 / 6, 5 / 6, 1])


def test_pu_example_reports_actual_capacity_and_custom_metric_uses_weights(metric_examples):
    ns = metric_examples
    values = ns["pu_values"]
    assert values["known_positive_capture_at_10pct"] == 0.5
    assert values["unlabeled_flagged_at_10pct"] == 0.5
    assert values["unlabeled_flagged_at_10pct"] > 0.1
    assert ns["metric"](ns["observed"], ns["probabilities"], ns["weight"]).to_dict()["mae"] == pytest.approx(2.1 / 6)


def test_data_examples_keep_alignment_and_source_values():
    namespace = {}
    for match in RECIPE.finditer((ROOT / "docs/api/data.md").read_text()):
        actual = execute(Recipe(match["name"], match["code"]), namespace)
        expected = (
            match["output"].split("```text\n", 1)[1].split("\n```", 1)[0]
            if match["output"] else ""
        )
        assert actual == expected, f"Stale data output: {match['name']}"
    data = namespace["data"]
    reordered = namespace["reordered"]
    assert reordered.y.to_list() == data.y.gather([2, 0, 2]).to_list()
    assert reordered.time.equals(data.time.gather([2, 0, 2]))
    assert namespace["age_only"].cat_features == ()
    assert namespace["transformed"].time.equals(data.time)
    assert not namespace["transformed"].y.equals(data.y)
    assert namespace["model_input"]["region"].tolist() == ["north", "unknown", "south"]
    assert namespace["with_missing"].X["region"].to_list() == ["north", None, "south"]
    np.testing.assert_allclose(namespace["native"].get_baseline().reshape(-1), [0.25] * 3)
