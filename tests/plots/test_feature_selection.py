"""Selection plots preserve actual search scores, choices and stopping reasons."""

from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pytest

from ml_toolkit.config import FigureConfig
from ml_toolkit.feature_selection import FeatureSelectionResult, FeatureSelectionStep
from ml_toolkit.plots import feature_selection as plots
from ml_toolkit.plots.utils import PlotError


def step(features, score, importances=None):
    return FeatureSelectionStep(
        tuple(features), {"score": score}, importances or {}, (), 0, 0, 0, None
    )


def history(kind, steps, selected, **settings):
    return FeatureSelectionResult(
        steps[selected].features,
        selected,
        tuple(steps),
        {"selector": kind, "metric": "score", "direction": "maximize", **settings},
    )


@pytest.fixture
def figures(monkeypatch):
    captured = []
    save = plots.save_figure

    def capture(fig, path, **kwargs):
        captured.append(fig)
        return save(fig, path, **kwargs)

    monkeypatch.setattr(plots, "save_figure", capture)
    return captured


@pytest.mark.parametrize("direction,sign", [("maximize", 1), ("minimize", -1)])
def test_rfe_real_scores_best_choice_and_tolerance(tmp_path, figures, direction, sign):
    scores = np.array([0.80, 0.818, 0.816, 0.79]) * sign
    result = history(
        "rfe",
        [step("abcdef"[:n], s) for n, s in zip([6, 4, 3, 1], scores)],
        2,
        direction=direction,
        score_tolerance=0.003,
    )
    plots.rfe_progress(result, tmp_path / "rfe.png")
    fig = figures[0]
    ax = fig.axes[0]
    np.testing.assert_equal(ax.lines[1].get_xdata(), [6, 4, 3, 1])
    np.testing.assert_equal(ax.lines[1].get_ydata(), scores)
    assert ax.get_xlim()[0] > ax.get_xlim()[1]
    np.testing.assert_allclose(ax.collections[1].get_offsets(), [[4, sign * 0.818]])
    np.testing.assert_allclose(ax.collections[2].get_offsets(), [[3, sign * 0.816]])
    ys = ax.collections[0].get_paths()[0].vertices[:, 1]
    np.testing.assert_allclose(sorted(set(ys)), sorted([sign * 0.818, sign * 0.815]))
    assert (
        "score" in ax.get_ylabel()
        and ("higher" if sign == 1 else "lower") in ax.get_ylabel()
    )
    assert "Selected: 3" in ax.texts[0].get_text()
    assert (tmp_path / "rfe.png").is_file() and not plt.fignum_exists(fig.number)


@pytest.mark.parametrize(
    "scores,selected", [([1.0], 0), ([1.0, 1.0], 1), ([1.0, 0.9], 0)]
)
def test_rfe_single_step_ties_and_best_selected_marker(
    tmp_path, figures, scores, selected
):
    result = history(
        "rfe",
        [step("ab"[: 2 - i], score) for i, score in enumerate(scores)],
        selected,
        score_tolerance=0,
    )
    plots.rfe_progress(result, tmp_path / "rfe.svg")
    ax = figures[0].axes[0]
    labels = ax.get_legend_handles_labels()[1]
    assert ("Best and selected" in labels) == (selected == 0)
    assert len(ax.collections) == (2 if selected == 0 else 3)


@pytest.mark.parametrize(
    "protected,selected", [((), "ab"), (("d",), "ad"), ((), "abcd")]
)
def test_topk_all_features_stable_ties_and_protection(
    tmp_path, figures, protected, selected
):
    result = history(
        "top_k",
        [step("abcd", 1, {"d": -0.1, "c": 0.8, "b": 0.8, "a": 0.9})],
        0,
        k=len(selected),
        protected_features=protected,
    )
    if selected != "abcd":
        result = replace(
            result,
            features=tuple(selected),
            selected_step=1,
            steps=(*result.steps, step(selected, 0)),
        )
    plots.top_k_ranking(result, tmp_path / "topk.png")
    ax = figures[0].axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "a",
        "b",
        "c",
        "d (protected)" if protected else "d",
    ]
    bars = {round(p.get_y() + p.get_height() / 2): p for p in ax.patches}
    assert [bars[i].get_width() for i in range(4)] == [0.9, 0.8, 0.8, -0.1]
    assert (bars[3].get_hatch() == "///") == bool(protected)
    if selected != "abcd":
        assert bars[0].get_facecolor() == bars[3 if protected else 1].get_facecolor()
        assert bars[0].get_facecolor() != bars[2].get_facecolor()
    assert bool(ax.lines) == (not protected and selected != "abcd")
    assert len(figures[0].axes) == 1


def test_topk_zero_importances_still_show_selection_status(tmp_path, figures):
    result = history(
        "top_k",
        [step("ab", 1, {"a": 0, "b": 0}), step("a", 1)],
        1,
        k=1,
    )
    plots.top_k_ranking(result, tmp_path / "zero.png")
    marks = figures[0].axes[0].collections
    np.testing.assert_equal(marks[0].get_offsets(), [[0, 0]])
    np.testing.assert_equal(marks[1].get_offsets(), [[0, 1]])
    assert not np.array_equal(marks[0].get_facecolors(), marks[1].get_facecolors())


def forward_history(direction="maximize", sign=1):
    return history(
        "forward",
        [
            step(names, score * sign)
            for names, score in [
                ("a", 0.6),
                ("b", 0.6),
                ("c", 0.5),
                ("ab", 0.7),
                ("ac", 0.69),
                ("abc", 0.701),
            ]
        ],
        3,
        direction=direction,
        min_improvement=0.01,
        max_features=3,
        max_fits=6,
    )


@pytest.mark.parametrize("direction,sign", [("maximize", 1), ("minimize", -1)])
def test_forward_all_candidates_path_ties_and_rejected_round(
    tmp_path, figures, direction, sign
):
    plots.forward_progress(forward_history(direction, sign), tmp_path / "forward.png")
    ax = figures[0].axes[0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1, 2])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([0.6, 0.7]) * sign)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets()[:, 1],
        np.array([0.6, 0.6, 0.5, 0.7, 0.69, 0.701]) * sign,
    )
    np.testing.assert_allclose(ax.collections[0].get_offsets()[[0, 3, 5], 0], [1, 2, 3])
    np.testing.assert_allclose(ax.collections[-1].get_offsets(), [[3, 0.701 * sign]])
    assert [t.get_text() for t in ax.texts] == ["+a", "+b", "Rejected: +c"]
    assert "improvement 0.001 ≤ min_improvement 0.01" in ax.get_title()
    # Accepted feature names stay below the title at the top of the plot.
    canvas = FigureCanvasAgg(figures[0])
    canvas.draw()
    renderer = canvas.get_renderer()
    assert all(t.get_window_extent(renderer).y1 < ax.bbox.y1 for t in ax.texts[:2])


@pytest.mark.parametrize(
    "max_features,max_fits,reason",
    [(2, 6, "max_features = 2"), (3, 5, "max_fits budget")],
)
def test_forward_stopping_at_limits_is_not_rejection(
    tmp_path, figures, max_features, max_fits, reason
):
    result = forward_history()
    result = replace(
        result,
        steps=result.steps[:-1],
        settings={
            **result.settings,
            "max_features": max_features,
            "max_fits": max_fits,
        },
    )
    plots.forward_progress(result, tmp_path / "limits.png")
    ax = figures[0].axes[0]
    assert reason in ax.get_title()
    assert "Best in rejected round" not in ax.get_legend_handles_labels()[1]


@pytest.mark.parametrize("baseline_only", [False, True])
def test_forward_protected_baseline_and_no_candidate_round(
    tmp_path, figures, baseline_only
):
    steps = [step("b", 1)]
    if not baseline_only:
        steps += [step("ab", 3), step("bc", 4)]
    result = history(
        "forward",
        steps,
        len(steps) - 1,
        min_improvement=0,
        protected_features=("b",),
        max_features=1 if baseline_only else 2,
        max_fits=3,
    )
    plots.forward_progress(result, tmp_path / "protected.png")
    ax = figures[0].axes[0]
    assert ax.texts[0].get_text() == "Protected baseline"
    if not baseline_only:
        np.testing.assert_equal(ax.lines[0].get_ydata(), [1, 4])
        assert ax.texts[1].get_text() == "+c"
    assert "max_features" in ax.get_title()


@pytest.mark.parametrize(
    "draw,kind",
    [
        (plots.rfe_progress, "rfe"),
        (plots.top_k_ranking, "top_k"),
        (plots.forward_progress, "forward"),
    ],
)
@pytest.mark.parametrize("format", ["png", "svg", "pdf"])
def test_export_formats_dimensions_dpi_and_closed_figures(
    tmp_path, figures, draw, kind, format
):
    if kind == "forward":
        result = forward_history()
    else:
        result = history(kind, [step("a", 1, {"a": 1})], 0, score_tolerance=0)
    before = plt.get_fignums()
    draw(
        result,
        tmp_path / "plot.png",
        figure=FigureConfig(format=format, figsize=(5, 4), dpi=72),
    )
    assert (tmp_path / f"plot.{format}").stat().st_size > 0
    np.testing.assert_equal(figures[0].get_size_inches(), [5, 4])
    assert plt.get_fignums() == before


@pytest.mark.parametrize(
    "draw,kind",
    [
        (plots.rfe_progress, "rfe"),
        (plots.top_k_ranking, "top_k"),
        (plots.forward_progress, "forward"),
    ],
)
def test_render_failure_closes_figure(tmp_path, monkeypatch, draw, kind):
    result = (
        forward_history()
        if kind == "forward"
        else history(kind, [step("a", 1, {"a": 1})], 0, score_tolerance=0)
    )

    def fail(*args, **kwargs):
        raise OSError("Cannot save figure")

    monkeypatch.setattr(plots, "save_figure", fail)
    before = plt.get_fignums()
    with pytest.raises(OSError, match="Cannot save"):
        draw(result, tmp_path / "error.png")
    assert plt.get_fignums() == before


def test_invalid_history_and_missing_scores_do_not_open_figures(tmp_path):
    result = history("rfe", [step("a", 1)], 0, score_tolerance=0)
    for invalid in [
        replace(result, steps=()),
        replace(result, selected_step=2),
        replace(result, steps=(step("a", float("nan")),)),
        replace(result, settings={**result.settings, "metric": "missing"}),
    ]:
        with pytest.raises(PlotError):
            plots.rfe_progress(invalid, tmp_path / "invalid.png")
    assert not plt.get_fignums() and not list(tmp_path.iterdir())
