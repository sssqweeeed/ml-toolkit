from dataclasses import dataclass

import numpy as np

from ..types import FloatArray


@dataclass(frozen=True)
class CumulativeCurve:
    cumulative_exposure: FloatArray
    cumulative_observed: FloatArray


@dataclass(frozen=True)
class SelectedRankingPoint:
    """Store gain and lift at a selected exposure percentage.

    Attributes
    ----------
    percent : float
        Exposure percentage in (0, 100].
    gain : float
        Interpolated fraction of total exposure-weighted observed outcome
        captured at this capacity.
    lift : float
        Gain divided by percent / 100.
    """

    percent: float
    gain: float
    lift: float


@dataclass(frozen=True)
class RankingMetrics:
    """Store an exposure-based ranking curve and its summary scores.

    Attributes
    ----------
    gini : float
        Twice the area under the gains curve minus one.
    normalized_gini : float
        Gini divided by perfect_gini; NaN when perfect_gini is close to zero.
    gains_area : float
        Area under the cumulative gains curve.
    perfect_gini : float
        Gini obtained by ranking rows by their observed target.
    cumulative_exposure : numpy.ndarray
        Cumulative exposure fractions, including the initial zero.
    cumulative_observed : numpy.ndarray
        Cumulative shares of exposure-weighted observed outcomes.
    cumulative_lift : numpy.ndarray
        Cumulative observed share divided by exposure share; NaN at zero
        exposure.
    selected_points : tuple[SelectedRankingPoint, ...]
        Interpolated gain and lift at each requested percentage, in request
        order. Each point exposes percent, gain, and lift attributes.
    """

    gini: float
    normalized_gini: float
    gains_area: float
    perfect_gini: float

    cumulative_exposure: FloatArray
    cumulative_observed: FloatArray
    cumulative_lift: FloatArray

    selected_points: tuple[SelectedRankingPoint, ...]


def _cumulative_curve(
    y_target: FloatArray,
    y_score: FloatArray,
    exposure: FloatArray,
) -> CumulativeCurve:
    y_target = np.asarray(y_target, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    if not (y_target.shape == y_score.shape == exposure.shape):
        raise ValueError("y_target, y_score and exposure must have the same shape")

    if np.any(~np.isfinite(y_target)):
        raise ValueError("y_target contains non-finite values")

    if np.any(~np.isfinite(y_score)):
        raise ValueError("y_score contains non-finite values")

    if np.any(~np.isfinite(exposure)):
        raise ValueError("exposure contains non-finite values")

    if np.any(exposure < 0):
        raise ValueError("exposure contains negative values")

    mask = exposure > 0

    y_target = y_target[mask]
    y_score = y_score[mask]
    exposure = exposure[mask]

    if len(y_target) == 0:
        raise ValueError("total_exposure <= 0")

    order = np.argsort(
        -y_score,
        kind="stable",
    )

    y_target = y_target[order]
    exposure = exposure[order]

    observed = y_target * exposure

    total_exposure = np.sum(exposure)
    total_observed = np.sum(observed)

    if total_exposure <= 0:
        raise ValueError("total_exposure <= 0")

    if total_observed <= 0:
        raise ValueError("total_observed <= 0")

    cumulative_exposure = np.concatenate(
        (
            [0.0],
            np.cumsum(exposure) / total_exposure,
        )
    )

    cumulative_observed = np.concatenate(
        (
            [0.0],
            np.cumsum(observed) / total_observed,
        )
    )

    return CumulativeCurve(
        cumulative_exposure=cumulative_exposure,
        cumulative_observed=cumulative_observed,
    )


def _gini_from_curve(
    curve: CumulativeCurve,
) -> tuple[float, float]:
    area = float(
        np.trapezoid(
            curve.cumulative_observed,
            curve.cumulative_exposure,
        )
    )

    gini = float(2.0 * area - 1.0)

    return gini, area


def _lift_from_curve(
    curve: CumulativeCurve,
) -> FloatArray:
    return np.divide(
        curve.cumulative_observed,
        curve.cumulative_exposure,
        out=np.full_like(
            curve.cumulative_observed,
            np.nan,
            dtype=float,
        ),
        where=curve.cumulative_exposure > 0,
    )


def _selected_ranking_points(
    curve: CumulativeCurve,
    selected_percents: tuple[float, ...],
) -> tuple[SelectedRankingPoint, ...]:
    result: list[SelectedRankingPoint] = []

    for percent in selected_percents:
        if not 0 < percent <= 100:
            raise ValueError(f"selected percent must be in (0, 100], got {percent}")

        share = percent / 100.0

        gain = float(
            np.interp(
                share,
                curve.cumulative_exposure,
                curve.cumulative_observed,
            )
        )

        lift = gain / share

        result.append(
            SelectedRankingPoint(
                percent=float(percent),
                gain=gain,
                lift=float(lift),
            )
        )

    return tuple(result)


def ranking_metrics(
    y_target: FloatArray,
    y_pred: FloatArray,
    exposure: FloatArray,
    selected_percents: tuple[float, ...] = (10.0,),
) -> RankingMetrics:
    """Calculate gains, lift, and Gini after ranking by descending prediction.

    Parameters
    ----------
    y_target : numpy.ndarray of shape (n_samples,)
        Finite observed rates with positive exposure-weighted total.
    y_pred : numpy.ndarray of shape (n_samples,)
        Finite scores used to order observations from highest to lowest.
    exposure : numpy.ndarray of shape (n_samples,)
        Finite, nonnegative row exposures with positive total.
    selected_percents : tuple[float, ...], default (10.0,)
        Exposure percentages in (0, 100] at which to interpolate gains and
        lift.

    Returns
    -------
    RankingMetrics
        Full cumulative curves, model and perfect Gini, and selected gain/lift
        points.

    Raises
    ------
    ValueError
        Shapes differ, inputs are non-finite, exposures are negative, totals
        are nonpositive, or a selected percentage is out of range.

    Notes
    -----
    Rows with zero exposure are omitted; score ties retain input order.
    Ranking capacity is measured by exposure, and observed amounts are
    y_target * exposure. Normalized Gini is NaN when
    the perfect-ranking Gini is close to zero.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_toolkit.metrics import ranking_metrics
    >>> result = ranking_metrics(
    ...     np.array([0.0, 1.0, 2.0]),
    ...     np.array([0.1, 0.5, 0.9]),
    ...     np.ones(3),
    ... )
    >>> round(result.normalized_gini, 2)
    1.0
    """
    model_curve = _cumulative_curve(
        y_target=y_target,
        y_score=y_pred,
        exposure=exposure,
    )

    gini, gains_area = _gini_from_curve(model_curve)

    perfect_curve = _cumulative_curve(
        y_target=y_target,
        y_score=y_target,
        exposure=exposure,
    )

    perfect_gini, _ = _gini_from_curve(perfect_curve)

    if np.isclose(perfect_gini, 0.0):
        normalized_gini = np.nan
    else:
        normalized_gini = gini / perfect_gini

    cumulative_lift = _lift_from_curve(model_curve)

    selected_points = _selected_ranking_points(
        curve=model_curve,
        selected_percents=selected_percents,
    )

    return RankingMetrics(
        gini=gini,
        normalized_gini=float(normalized_gini),
        gains_area=gains_area,
        perfect_gini=perfect_gini,
        cumulative_exposure=model_curve.cumulative_exposure,
        cumulative_observed=model_curve.cumulative_observed,
        cumulative_lift=cumulative_lift,
        selected_points=selected_points,
    )


ranking_metrics = ranking_metrics
