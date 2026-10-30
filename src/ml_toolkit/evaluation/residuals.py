"""Weighted regression residual tables and prediction-report integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl

from .base import PredictionEvaluator
from ..config import FigureConfig
from ..data import Pool
from ..data_analysis._common import validate_split
from ..feature_levels import FeatureLevelConfig
from ..metrics._evaluation import validate_prediction
from ..types import FloatArray, IntArray

ResidualLoss = Literal["squared", "absolute", "poisson", "gamma"]


@dataclass(frozen=True)
class ResidualResult:
    """Keep aligned observations, weighted bin summaries, and worst rows.

    Attributes
    ----------
    rows : polars.DataFrame
        Every input row in its original order, including zero-weight rows.
        Contains row_index, split, time, observed, prediction, weight, residual,
        loss, contribution, and prediction_bin, plus object_id when supplied.
        contribution is row weight multiplied by the nonnegative unit loss.
    bins : polars.DataFrame
        Positive-weight summaries for each populated split/prediction-bin pair.
        Bin bounds use positive-weight train predictions when available;
        otherwise all supplied positive-weight predictions define the bins.
        Summaries contain counts, total weight, weighted means, and the inverse
        weighted empirical CDF residual quantiles p05, p25, p50, p75, and p95.
    worst : polars.DataFrame
        Up to max_rows positive-weight observations per supplied split, ordered
        by decreasing contribution and then original row_index. It has the
        same schema as rows. This table includes targets and optional IDs.
    """

    rows: pl.DataFrame
    bins: pl.DataFrame
    worst: pl.DataFrame


def _quantiles(
    values: FloatArray, weight: FloatArray, probabilities: FloatArray
) -> FloatArray:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weight[order])
    positions = cast(
        IntArray,
        cumulative.searchsorted(probabilities * cumulative[-1], side="left"),
    )
    return values[order[np.minimum(positions, len(order) - 1)]]


def _losses(
    observed: FloatArray, prediction: FloatArray, loss: ResidualLoss
) -> tuple[FloatArray, FloatArray]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = prediction - observed
        if loss == "absolute":
            unit = np.abs(difference)
            residual = difference
        elif loss == "squared":
            unit = difference**2
            residual = difference
        else:
            if (prediction <= 0).any():
                raise ValueError(
                    f"{loss} residuals require strictly positive predictions"
                )
            if loss == "poisson":
                if (observed < 0).any():
                    raise ValueError("poisson residuals require nonnegative targets")
                unit = 2 * prediction.copy()
                positive = observed > 0
                y, mu = observed[positive], prediction[positive]
                ratio = (y - mu) / mu
                log_ratio = np.log(y) - np.log(mu)
                close = np.abs(ratio) < 0.25
                log_ratio[close] = np.log1p(ratio[close])
                positive_unit = 2 * (y * log_ratio - (y - mu))
                near = np.abs(ratio) < 1e-3
                r = ratio[near]
                # 2 * ((1+r) * log1p(r) - r) / r**2, evaluated
                # without subtracting its two almost equal first-order terms.
                factor = 1 + r * (
                    -1 / 3
                    + r
                    * (1 / 6 + r * (-1 / 10 + r * (1 / 15 + r * (-1 / 21 + r / 28))))
                )
                positive_unit[near] = (mu[near] * r) * r * factor
                unit[positive] = positive_unit
            else:
                if (observed <= 0).any():
                    raise ValueError(
                        "gamma residuals require strictly positive targets"
                    )
                ratio = (observed - prediction) / prediction
                unit = 2 * (
                    observed / prediction - 1 + np.log(prediction) - np.log(observed)
                )
                close = np.abs(ratio) < 0.25
                unit[close] = 2 * (ratio[close] - np.log1p(ratio[close]))
                near = np.abs(ratio) < 1e-3
                r = ratio[near]
                # 2 * (r - log1p(r)) / r**2 has a well-conditioned
                # Taylor expansion at zero, including exact perfect fits.
                factor = 1 + r * (
                    -2 / 3
                    + r * (1 / 2 + r * (-2 / 5 + r * (1 / 3 + r * (-2 / 7 + r / 4))))
                )
                unit[near] = r * r * factor
            unit = np.maximum(unit, 0)
            residual = np.sign(difference) * np.sqrt(unit)
    if not np.isfinite(residual).all() or not np.isfinite(unit).all():
        raise ValueError(
            "Residuals and unit losses must be finite; values exceed the numeric range"
        )
    return residual, unit


@dataclass(frozen=True, kw_only=True)
class Residuals(PredictionEvaluator):
    """Inspect weighted errors, prediction-dependent bias, and large losses.

    Parameters
    ----------
    name : str, default 'residuals'
        Unique report name and artifact namespace.
    loss : {'squared', 'absolute', 'poisson', 'gamma'}, default 'squared'
        Loss used to rank individual weighted contributions. Squared/absolute
        use prediction minus target as residual. Poisson/Gamma use the signed
        square root of their unit deviance with the same sign convention.
        Poisson requires nonnegative targets and positive predictions; Gamma
        requires both targets and predictions to be strictly positive.
    n_bins : int, default 10
        Maximum number of weighted prediction-quantile bins. Repeated cutoffs
        collapse. Positive-weight train rows define cutoffs when available;
        held-out-only calls use their supplied positive-weight rows.
    histogram_bins : int, default 40
        Number of common residual histogram bins across supplied splits.
    max_samples : int, default 10000
        Maximum uniformly sampled positive-weight scatter points per split.
        Does not change tables, histogram weights, or quantile calculations.
    max_rows : int, default 20
        Maximum largest-contribution observations exported for each split.
    seed : int, default 0
        Nonnegative seed used only for scatter sampling.

    Notes
    -----
    Includes every supplied split and supports train-only reports. Zero-weight
    rows remain in rows.parquet but do not enter summaries, figures, or worst
    rows. Exports rows.parquet, bins.parquet, worst.parquet and residuals.<format>.
    Rows and worst tables contain observed outcomes and optional object IDs.
    The report retains settings only, never changes inputs or refits a model.
    """

    name: str = "residuals"
    loss: ResidualLoss = "squared"
    n_bins: int = 10
    histogram_bins: int = 40
    max_samples: int = 10_000
    max_rows: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.loss not in ("squared", "absolute", "poisson", "gamma"):
            raise ValueError("loss must be squared, absolute, poisson, or gamma")
        for label, value in (
            ("n_bins", self.n_bins),
            ("histogram_bins", self.histogram_bins),
            ("max_samples", self.max_samples),
            ("max_rows", self.max_rows),
        ):
            if value < 1:
                raise ValueError(f"{label} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    def parameters(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "n_bins": self.n_bins,
            "histogram_bins": self.histogram_bins,
            "max_samples": self.max_samples,
            "max_rows": self.max_rows,
            "seed": self.seed,
        }

    def calculate(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
    ) -> ResidualResult:
        """Calculate aligned residuals and complete weighted summaries.

        Parameters
        ----------
        data : Pool
            Aligned observations with finite targets and finite nonnegative
            weights with positive total. All rows must satisfy the loss domain.
        prediction : FloatArray
            One finite prediction per row in the original target units.
        split : polars.Series
            Non-null String labels containing train and at most one of val/test;
            held-out-only data is also supported.

        Returns
        -------
        ResidualResult
            Original-order row details, weighted prediction-bin statistics,
            and largest weighted loss contributions per split.
        """
        validate_prediction(data, prediction)
        validate_split(data, split)
        observed = data.y.to_numpy().astype(float)
        weight = data.weight.to_numpy().astype(float)
        if not np.isfinite(observed).all():
            raise ValueError("Targets must be finite")
        if (
            not np.isfinite(weight).all()
            or (weight < 0).any()
            or not (weight > 0).any()
        ):
            raise ValueError(
                "Weights must be finite, nonnegative, and have positive total"
            )
        residual, unit = _losses(observed, prediction, self.loss)
        with np.errstate(over="ignore", invalid="ignore"):
            contribution = unit * weight
        if (
            not np.isfinite(contribution).all()
            or not np.isfinite(contribution.sum())
            or not np.isfinite(weight.sum())
        ):
            raise ValueError("Weighted contributions and total weight must be finite")
        positive = weight > 0
        reference = positive & (split == "train").to_numpy()
        if not reference.any():
            reference = positive
        reference_prediction = prediction[reference]
        boundaries = np.unique(
            _quantiles(
                reference_prediction,
                weight[reference],
                np.arange(1, self.n_bins, dtype=float) / self.n_bins,
            )
        )
        boundaries = boundaries[
            (boundaries > reference_prediction.min())
            & (boundaries < reference_prediction.max())
        ]
        bin_ids = np.searchsorted(boundaries, prediction, side="right")
        rows = pl.DataFrame(
            {
                "row_index": np.arange(len(data), dtype=np.int64),
                "split": split,
                "time": data.time,
                "observed": observed,
                "prediction": prediction,
                "weight": weight,
                "residual": residual,
                "loss": unit,
                "contribution": contribution,
                "prediction_bin": bin_ids.astype(np.int64),
            }
        )
        if data.object_id is not None:
            rows = rows.with_columns(data.object_id.alias("object_id"))
        lower = np.concatenate(([-np.inf], boundaries))
        upper = np.concatenate((boundaries, [np.inf]))
        records: list[dict[str, object]] = []
        worst: list[pl.DataFrame] = []
        for part in ("train", "val", "test"):
            mask = (split == part).to_numpy() & positive
            if not mask.any():
                continue
            selected = rows.filter(pl.Series(mask))
            worst.append(
                selected.sort(
                    ["contribution", "row_index"], descending=[True, False]
                ).head(self.max_rows)
            )
            for bin_id in np.unique(bin_ids[mask]):
                indices = np.flatnonzero(mask & (bin_ids == bin_id))
                current_weight = weight[indices]
                quantiles = _quantiles(
                    residual[indices],
                    current_weight,
                    np.asarray([0.05, 0.25, 0.5, 0.75, 0.95]),
                )
                records.append(
                    {
                        "split": part,
                        "prediction_bin": int(bin_id),
                        "lower_bound": float(lower[bin_id]),
                        "upper_bound": float(upper[bin_id]),
                        "n_rows": len(indices),
                        "sum_weight": float(current_weight.sum()),
                        "mean_prediction": float(
                            np.dot(
                                prediction[indices],
                                current_weight / current_weight.sum(),
                            )
                        ),
                        "mean_residual": float(
                            np.average(residual[indices], weights=current_weight)
                        ),
                        "mean_loss": float(
                            np.average(unit[indices], weights=current_weight)
                        ),
                        "sum_contribution": float(contribution[indices].sum()),
                        **dict(
                            zip(
                                ("p05", "p25", "p50", "p75", "p95"),
                                quantiles.tolist(),
                                strict=True,
                            )
                        ),
                    }
                )
        schema = {
            "split": pl.String,
            "prediction_bin": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound": pl.Float64,
            "n_rows": pl.Int64,
            "sum_weight": pl.Float64,
            "mean_prediction": pl.Float64,
            "mean_residual": pl.Float64,
            "mean_loss": pl.Float64,
            "sum_contribution": pl.Float64,
            **{name: pl.Float64 for name in ("p05", "p25", "p50", "p75", "p95")},
        }
        return ResidualResult(
            rows=rows,
            bins=pl.DataFrame(records, schema=schema),
            worst=pl.concat(worst) if worst else rows.clear(),
        )

    def write_artifacts(
        self,
        *,
        data: Pool,
        prediction: FloatArray,
        split: pl.Series,
        artifact_dir: Path,
        figure: FigureConfig,
        features: tuple[FeatureLevelConfig, ...] | None,
        n_bins: int,
    ) -> list[Path]:
        from ..plots.residuals import residual_diagnostics

        result = self.calculate(data=data, prediction=prediction, split=split)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for name, table in (
            ("rows", result.rows),
            ("bins", result.bins),
            ("worst", result.worst),
        ):
            path = artifact_dir / f"{name}.parquet"
            table.write_parquet(path)
            paths.append(path)
        path = artifact_dir / f"residuals.{figure.format}"
        residual_diagnostics(
            result,
            save_fig=path,
            loss=self.loss,
            histogram_bins=self.histogram_bins,
            max_samples=self.max_samples,
            seed=self.seed,
            figure=figure,
        )
        paths.append(path)
        return paths
