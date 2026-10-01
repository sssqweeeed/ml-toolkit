from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class CumulativeCurve:
    cumulative_exposure: np.ndarray
    cumulative_observed: np.ndarray

@dataclass(frozen=True)
class SelectedRankingPoint:
    percent: float
    gain: float
    lift: float

@dataclass(frozen=True)
class RankingMetrics:
    gini: float
    normalized_gini: float
    gini_area: float
    perfect_gini: float

    cumulative_exposure: np.ndarray
    cumulative_observed: np.ndarray
    cumulative_lift: np.ndarray

    selected_points: tuple[SelectedRankingPoint, ...]

def _cumulative_curve(
    y_target: np.ndarray,
    y_score: np.ndarray,
    exposure: np.ndarray,
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
) -> np.ndarray:
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
    result = []

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

def _ranking_metrics(
    y_target: np.ndarray,
    y_pred: np.ndarray,
    exposure: np.ndarray,
    selected_percents: tuple[float, ...] = (10.0,),
) -> RankingMetrics:
    model_curve = _cumulative_curve(
        y_target=y_target,
        y_score=y_pred,
        exposure=exposure,
    )

    gini, gini_area = _gini_from_curve(model_curve)

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
        gini_area=gini_area,
        perfect_gini=perfect_gini,
        cumulative_exposure=model_curve.cumulative_exposure,
        cumulative_observed=model_curve.cumulative_observed,
        cumulative_lift=cumulative_lift,
        selected_points=selected_points,
    )

ranking_metrics = _ranking_metrics
