"""Typed selection histories serialize to the flat JSON logged by 6.x."""

import json
from dataclasses import replace

from ml_toolkit.feature_selection import (
    CustomSelectionSettings,
    FeatureSelectionResult,
    FeatureSelectionStep,
    ForwardSettings,
    RFESettings,
    TopKSettings,
)
from ml_toolkit.feature_selection.history import forward_rounds, forward_stop


def step(features, score):
    return FeatureSelectionStep(
        tuple(features), {"score": score}, {}, (), 0.0, 0.0, 0.0, None
    )


def payload(settings):
    result = FeatureSelectionResult(("a",), 0, (step("a", 1.0),), settings)
    return json.loads(json.dumps(result.to_dict()))


def test_rfe_settings_match_legacy_json_including_key_order():
    settings = RFESettings(
        min_features=1,
        metric="mae",
        direction="minimize",
        max_fits=6,
        score_tolerance=0.0,
        protected_features=("a",),
    )
    saved = payload(settings)["settings"]
    assert saved == {
        "selector": "rfe",
        "min_features": 1,
        "metric": "mae",
        "direction": "minimize",
        "max_fits": 6,
        "score_tolerance": 0.0,
        "protected_features": ["a"],
    }
    assert list(saved) == [
        "selector",
        "min_features",
        "metric",
        "direction",
        "max_fits",
        "score_tolerance",
        "protected_features",
    ]
    named = payload(replace(settings, evaluator="business"))
    assert list(named["settings"])[-1] == "evaluator"
    assert named["settings"]["evaluator"] == "business"


def test_top_k_forward_and_custom_settings_match_legacy_json():
    assert payload(TopKSettings(k=2, protected_features=()))["settings"] == {
        "selector": "top_k",
        "k": 2,
        "protected_features": [],
    }
    forward = ForwardSettings(
        metric="score",
        direction="maximize",
        max_features=3,
        max_fits=6,
        min_improvement=0.5,
        protected_features=(),
        evaluator="business",
    )
    assert list(payload(forward)["settings"]) == [
        "selector",
        "metric",
        "direction",
        "max_features",
        "max_fits",
        "min_improvement",
        "protected_features",
        "evaluator",
    ]
    custom = CustomSelectionSettings("fixed", {"features": ("a", "b")})
    assert payload(custom)["settings"] == {"selector": "fixed", "features": ["a", "b"]}
    assert payload(None)["settings"] == {}


def test_history_top_level_keys_and_steps_match_legacy_layout():
    saved = payload(None)
    assert list(saved) == ["features", "selected_step", "steps", "settings"]
    assert list(saved["steps"][0]) == [
        "features",
        "metrics",
        "importances",
        "eliminated_features",
        "fit_seconds",
        "evaluation_seconds",
        "ranking_seconds",
        "best_iteration",
        "objective_metrics",
    ]


def test_forward_rounds_and_stop_reason():
    steps = tuple(
        step(names, score)
        for names, score in [("a", 0.6), ("b", 0.5), ("ab", 0.7), ("abc", 0.71)]
    )
    settings = ForwardSettings(
        metric="score",
        direction="maximize",
        max_features=3,
        max_fits=9,
        min_improvement=0.05,
        protected_features=(),
    )
    result = FeatureSelectionResult(("a", "b"), 2, steps[:3], settings)
    rounds = forward_rounds(result, settings)
    assert [(r.count, r.best, r.accepted) for r in rounds] == [
        (1, 0, True),
        (2, 2, True),
    ]
    assert forward_stop(result, settings, rounds).reason == "max_fits"
    capped = replace(settings, max_features=2)
    assert forward_stop(result, capped, rounds).reason == "max_features"
