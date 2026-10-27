"""Feature selection diagnostics drawn from stored histories without model fits."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import FigureConfig
from ..feature_selection import FeatureSelectionResult
from ..types import FloatArray
from .utils import PlotError, numeric_vector, save_figure

__all__ = ["rfe_progress", "top_k_ranking", "forward_progress"]


def _validate_result(result: FeatureSelectionResult, selector: str) -> None:
    if result.settings.get("selector") != selector:
        raise PlotError(f"Expected a {selector!r} selection history")
    if (
        not result.steps
        or not 0 <= result.selected_step < len(result.steps)
        or not result.features
        or result.steps[result.selected_step].features != result.features
    ):
        raise PlotError("selected_step must identify the selected nonempty subset")


def _objective(result: FeatureSelectionResult) -> tuple[str, bool, FloatArray]:
    metric = result.settings.get("metric")
    direction = result.settings.get("direction")
    if not isinstance(metric, str) or not metric.strip():
        raise PlotError("Selection settings must contain a metric name")
    if direction not in ("minimize", "maximize"):
        raise PlotError("Selection direction must be minimize or maximize")
    try:
        scores = numeric_vector([step.metrics[metric] for step in result.steps], metric)
    except KeyError as exc:
        raise PlotError(f"Selection metric {metric!r} is missing from history") from exc
    return metric, direction == "maximize", scores


def _number(result: FeatureSelectionResult, name: str) -> float:
    value = result.settings.get(name)
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise PlotError(f"Selection setting {name!r} must be finite and nonnegative")
    return float(value)


def _protected(result: FeatureSelectionResult) -> tuple[str, ...]:
    value = result.settings.get("protected_features", ())
    if not isinstance(value, tuple):
        raise PlotError("protected_features must be a tuple of feature names")
    return value


def rfe_progress(
    result: FeatureSelectionResult,
    save_fig: str | Path,
    *,
    figure: FigureConfig | None = None,
) -> None:
    """Save validation quality against decreasing RFE subset size.

    Parameters
    ----------
    result : FeatureSelectionResult
        Stored RFE history, including objective, direction and score_tolerance.
    save_fig : str or pathlib.Path
        Destination figure path; parent directories are created.
    figure : FigureConfig or None, default None
        Shared export size, DPI and format; None preserves native dimensions.

    Notes
    -----
    Marks the best and selected subsets and the absolute tolerance band. Scores
    describe the selection search, not independent held-out performance.
    Creates no models and closes the figure even when rendering fails.
    """
    _validate_result(result, "rfe")
    metric, maximize, scores = _objective(result)
    tolerance = _number(result, "score_tolerance")
    counts = [len(step.features) for step in result.steps]
    best = int(np.argmax(scores) if maximize else np.argmin(scores))
    chosen = result.selected_step
    lower, upper = (
        (scores[best] - tolerance, scores[best])
        if maximize
        else (scores[best], scores[best] + tolerance)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        ax.fill_between(
            [max(counts) + 0.5, min(counts) - 0.5],
            lower,
            upper,
            color="C0",
            alpha=0.12,
            label=f"Within tolerance ({tolerance:g})",
        )
        ax.axhline(scores[best], color="0.5", linewidth=1, linestyle=":")
        ax.plot(counts, scores, "o-", color="C0", label="Evaluated subset")
        ax.scatter(
            [counts[best]],
            [scores[best]],
            marker="D",
            color="0.2",
            s=65,
            label="Best and selected" if best == chosen else "Best score",
            zorder=4,
        )
        if best != chosen:
            ax.scatter(
                [counts[chosen]],
                [scores[chosen]],
                facecolors="none",
                edgecolors="C0",
                linewidths=2,
                s=180,
                label="Selected subset",
                zorder=5,
            )
        ax.annotate(
            f"Selected: {counts[chosen]} features\n{metric} = {scores[chosen]:.6g}",
            (counts[chosen], scores[chosen]),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
        ax.set(
            xlabel="Number of features (selection proceeds left to right)",
            ylabel=f"Validation {metric} ({'↑ higher' if maximize else '↓ lower'} is better)",
            title="RFE feature selection",
        )
        ax.set_xticks(counts)
        ax.set_xlim(max(counts) + 0.5, min(counts) - 0.5)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)


def top_k_ranking(
    result: FeatureSelectionResult,
    save_fig: str | Path,
    *,
    figure: FigureConfig | None = None,
) -> None:
    """Save the full-model ranking with selected and protected features marked.

    Parameters
    ----------
    result : FeatureSelectionResult
        Top-K history whose first step stores full-model ranking values.
    save_fig : str or pathlib.Path
        Destination figure path; parent directories are created.
    figure : FigureConfig or None, default None
        Shared export settings. The default height adapts to all input features.

    Notes
    -----
    Ties preserve input order. Protected features are mandatory and may be below
    the unprotected ranking cutoff, so no global Top-K line is drawn for them.
    Uses raw ranking values, creates no models, and always closes the figure.
    """
    _validate_result(result, "top_k")
    full = result.steps[0]
    if not full.features or set(full.importances) != set(full.features):
        raise PlotError("Top-K history must contain all full-model importances")
    numeric_vector([full.importances[name] for name in full.features], "importance")
    names = sorted(full.features, key=full.importances.__getitem__, reverse=True)
    protected = set(_protected(result))
    chosen = set(result.features)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(names) + 2)))
    try:
        for label, color, hatch, positions in (
            (
                "Selected",
                "C0",
                "",
                [i for i, n in enumerate(names) if n in chosen - protected],
            ),
            (
                "Excluded",
                "0.8",
                "",
                [i for i, n in enumerate(names) if n not in chosen],
            ),
            (
                "Protected (selected)",
                "C0",
                "///",
                [i for i, n in enumerate(names) if n in protected],
            ),
        ):
            if positions:
                ax.barh(
                    positions,
                    [full.importances[names[i]] for i in positions],
                    color=color,
                    hatch=hatch,
                    edgecolor="0.3",
                    linewidth=0,
                    label=label,
                )
                zero_positions = [
                    i for i in positions if full.importances[names[i]] == 0
                ]
                if zero_positions:
                    # A zero-length bar must still communicate selection status.
                    ax.scatter(
                        [0] * len(zero_positions),
                        zero_positions,
                        color=color,
                        s=24,
                        zorder=3,
                    )
        if not protected and len(chosen) < len(names):
            ax.axhline(
                len(chosen) - 0.5,
                color="0.5",
                linestyle="--",
                linewidth=1,
                label=f"Top-{len(chosen)} cutoff",
            )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(
            [f"{name} (protected)" if name in protected else name for name in names]
        )
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set(
            xlabel="Ranking importance (raw units)",
            ylabel="Feature",
            title=f"Top-K feature selection: {len(chosen)} of {len(names)} selected",
        )
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)


def forward_progress(
    result: FeatureSelectionResult,
    save_fig: str | Path,
    *,
    figure: FigureConfig | None = None,
) -> None:
    """Save all Forward candidates, accepted additions and the stopping reason.

    Parameters
    ----------
    result : FeatureSelectionResult
        Complete Forward history, including rejected candidates and settings.
    save_fig : str or pathlib.Path
        Destination figure path; parent directories are created.
    figure : FigureConfig or None, default None
        Shared export dimensions, DPI and format.

    Notes
    -----
    Consecutive candidate subsets of equal size form one complete round. Its
    best candidate is accepted only through the returned subset's size; a later
    round is rejected. Equal scores retain the first candidate. A protected-only
    baseline is distinct from additions. No models or predictions are created.
    """
    _validate_result(result, "forward")
    metric, maximize, scores = _objective(result)
    threshold = _number(result, "min_improvement")
    max_features = _number(result, "max_features")
    max_fits = _number(result, "max_fits")
    protected = _protected(result)
    counts = [len(step.features) for step in result.steps]
    groups = [
        (size, list(indices))
        for size, indices in groupby(range(len(result.steps)), key=counts.__getitem__)
    ]
    # Reconstruct accepted winners from the model-free, chronological history.
    accepted: list[int] = []
    labels: list[str] = []
    candidate_x: list[float] = []
    candidate_y: list[float] = []
    rejected: int | None = None
    previous = set(protected)
    for size, indices in groups:
        if protected and size == len(protected):
            accepted.append(indices[0])
            labels.append("Protected baseline")
            continue
        best = (max if maximize else min)(indices, key=lambda i: scores[i])
        # Keep winners on the round's center so highlights cover their marks.
        # Other candidates use deterministic jitter, including exact score ties.
        spread = np.linspace(-0.18, 0.18, len(indices))
        offsets = iter(np.delete(spread, int(np.argmin(np.abs(spread)))))
        for index in indices:
            candidate_x.append(size + (0.0 if index == best else float(next(offsets))))
            candidate_y.append(float(scores[index]))
        if size <= len(result.features):
            accepted.append(best)
            added = [n for n in result.steps[best].features if n not in previous]
            labels.append("+" + ", ".join(added))
            previous = set(result.steps[best].features)
        else:
            rejected = best
    if not accepted or accepted[-1] != result.selected_step:
        raise PlotError("Forward history must identify its accepted round winners")
    if rejected is not None:
        improvement = float(scores[rejected] - scores[result.selected_step])
        if not maximize:
            improvement = -improvement
        stop = f"Stopped: improvement {improvement:.6g} ≤ min_improvement {threshold:g}"
    elif len(result.features) >= max_features:
        stop = f"Stopped: max_features = {max_features:g}"
    else:
        stop = f"Stopped: max_fits budget ({len(result.steps)}/{max_fits:g} fits; next full round does not fit)"
    fig, ax = plt.subplots(figsize=(11, 6))
    try:
        if candidate_x:
            ax.scatter(
                candidate_x,
                candidate_y,
                color="0.65",
                alpha=0.6,
                s=28,
                label="Evaluated candidate",
            )
        ax.plot(
            [counts[i] for i in accepted],
            [scores[i] for i in accepted],
            "o-",
            color="C0",
            label="Accepted path",
            zorder=3,
        )
        for index, label in zip(accepted, labels):
            ax.annotate(
                label,
                (counts[index], scores[index]),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        chosen = result.selected_step
        ax.scatter(
            [counts[chosen]],
            [scores[chosen]],
            facecolors="none",
            edgecolors="C0",
            linewidths=2,
            s=160,
            label="Selected subset",
            zorder=4,
        )
        if rejected is not None:
            ax.scatter(
                [counts[rejected]],
                [scores[rejected]],
                marker="D",
                color="C1",
                s=65,
                label="Best in rejected round",
                zorder=4,
            )
            added = [
                n for n in result.steps[rejected].features if n not in result.features
            ]
            ax.annotate(
                "Rejected: +" + ", ".join(added),
                (counts[rejected], scores[rejected]),
                xytext=(-8, -28),
                textcoords="offset points",
                ha="right",
                fontsize=9,
            )
        ax.set(
            xlabel="Number of features (selection rounds)",
            ylabel=f"Validation {metric} ({'↑ higher' if maximize else '↓ lower'} is better)",
            title=f"Forward feature selection\n{stop}",
        )
        ax.set_xticks([size for size, _ in groups])
        ax.set_xlim(min(counts) - 0.5, max(counts) + 0.5)
        # Reserve room above the best candidates for feature-name annotations.
        span = float(np.ptp(scores)) or max(abs(float(scores[0])) * 0.05, 0.01)
        ax.set_ylim(
            float(np.min(scores)) - span * 0.08, float(np.max(scores)) + span * 0.18
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, save_fig, config=figure)
    finally:
        plt.close(fig)
