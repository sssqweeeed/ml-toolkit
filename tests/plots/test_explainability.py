import pytest

from ml_toolkit.plots.explainability import feature_importance


def test_feature_importance_sorts_and_limits_displayed_features(tmp_path):
    output = tmp_path / "nested" / "importance.png"
    ordered = feature_importance(
        ["a", "b", "c"], [0.2, 0.8, 0.4], output, max_features_display=2
    )
    assert ordered.feature.tolist() == ["b", "c"]
    assert ordered.importance.tolist() == [0.8, 0.4]
    assert output.stat().st_size > 0


@pytest.mark.parametrize(
    "names,importance,limit,message",
    [
        (["a"], [0.2, 0.8], 2, "length"),
        (["a"], [float("nan")], 2, "finite"),
        (["a"], [0.2], 0, "positive"),
    ],
)
def test_feature_importance_rejects_invalid_inputs(
    names, importance, limit, message, tmp_path
):
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match=message):
        feature_importance(names, importance, output, max_features_display=limit)
    assert not output.exists()
