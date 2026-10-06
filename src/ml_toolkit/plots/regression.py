"""Observed-versus-fitted diagnostics for continuous targets."""

from pathlib import Path

from ..config import FigureConfig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import PlotError, numeric_vector, same_length, save_figure, weights


def _aggregate(
    groups, observed, fitted, weight, name: str, *, sort: bool
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            name: groups,
            "observed_weighted": observed * weight,
            "fitted_weighted": fitted * weight,
            "volume": weight,
        }
    )
    grouped = frame.groupby(name, sort=sort, as_index=False, observed=True).sum(
        numeric_only=True
    )
    grouped = grouped.loc[grouped.volume > 0].copy()
    grouped["observed"] = grouped.observed_weighted / grouped.volume
    grouped["fitted"] = grouped.fitted_weighted / grouped.volume
    return grouped[[name, "observed", "fitted", "volume"]]


def observed_fitted(
    feature,
    observed,
    fitted,
    save_fig: str | Path,
    *,
    weight=None,
    feature_name: str = "Feature",
    n_bins: int = 10,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Plot weighted observed and fitted means by feature group.

        Parameters
        ----------
        feature : array-like of shape (n_samples,)
            Feature values. Varying numeric features are quantile-binned;
            categorical and constant features are grouped by value. Missing values
            form a missing group.
        observed : array-like of shape (n_samples,)
            Nonempty finite observed values.
        fitted : array-like of shape (n_samples,)
            Finite predictions aligned with observations.
        save_fig : str or pathlib.Path
            Output figure path; parent directories are created.
        weight : array-like or None, default None
            Finite, nonnegative exposure weights with positive total; None uses row
            counts.
        feature_name : str, default 'Feature'
            Feature label used on the plot.
        n_bins : int, default 10
            Positive requested number of numeric quantile bins; duplicate edges are
            dropped.

        figure : FigureConfig or None, default None
            Optional export dimensions, DPI, and format. None preserves native
            dimensions and the requested path format.

        Returns
        -------
        pandas.DataFrame
            Columns feature, observed, fitted, and volume. Means are weighted by
            exposure; volume is group total weight. Zero-weight groups are omitted.

        Raises
        ------
        PlotError
            Values, lengths, or exposure weights are invalid.
        ValueError
            n_bins is nonpositive.

        Notes
        -----
        Creates parent directories for save_fig and closes the figure after saving.

        Examples
        --------
        >>> from pathlib import Path

    from ..config import FigureConfig
        >>> from tempfile import TemporaryDirectory
        >>> from ml_toolkit.plots.regression import observed_fitted
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "observed.png"
        ...     table = observed_fitted(["a", "a"], [1.0, 3.0], [2.0, 2.0], path)
        ...     print(table["observed"].tolist(), path.is_file())
        [2.0] True
    """
    y, prediction = numeric_vector(observed, "observed"), numeric_vector(
        fitted, "fitted"
    )
    values = pd.Series(feature, name=feature_name)
    same_length(feature=values, observed=y, fitted=prediction)
    exposure = weights(weight, len(y))
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    numeric = pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 1
    if numeric:
        groups = pd.qcut(
            values, q=min(n_bins, values.nunique(dropna=True)), duplicates="drop"
        )
        groups = groups.astype("string").fillna("missing")
    else:
        groups = values.astype("string").fillna("missing")
    grouped = _aggregate(groups, y, prediction, exposure, "feature", sort=numeric)
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(10, 6))
    volume_ax = ax.twinx()
    volume_ax.bar(x, grouped.volume, alpha=0.2, color="tab:gray")
    ax.plot(x, grouped.observed, marker="o", label="Observed")
    ax.plot(x, grouped.fitted, marker="o", label="Fitted")
    ax.set_xticks(x, grouped.feature.astype(str), rotation=45, ha="right")
    ax.set(xlabel=feature_name, ylabel="Observed / fitted")
    volume_ax.set_ylabel("Exposure / count")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped


def monthly_observed_fitted(
    reference_timestamp,
    reference_observed,
    reference_fitted,
    current_timestamp,
    current_observed,
    current_fitted,
    save_fig: str | Path,
    *,
    reference_weight=None,
    current_weight=None,
    reference_group=None,
    current_group=None,
    figure: FigureConfig | None = None,
) -> pd.DataFrame:
    """Plot monthly weighted means for two samples and optional groups.

    Parameters
    ----------
    reference_timestamp : array-like
        Datetime-like values aligned with reference observations.
    reference_observed : array-like
        Nonempty finite reference targets.
    reference_fitted : array-like
        Finite reference predictions aligned with targets.
    current_timestamp : array-like
        Datetime-like values aligned with current observations.
    current_observed : array-like
        Nonempty finite current targets.
    current_fitted : array-like
        Finite current predictions aligned with targets.
    save_fig : str or pathlib.Path
        Output figure path; parent directories are created.
    reference_weight : array-like or None, default None
        Finite, nonnegative reference weights with positive total; None uses
        equal weights.
    current_weight : array-like or None, default None
        Finite, nonnegative current weights with positive total; None uses
        equal weights.
    reference_group : array-like or None, default None
        Optional reference group labels; None groups all rows as all.
    current_group : array-like or None, default None
        Optional current group labels; None groups all rows as all.

    figure : FigureConfig or None, default None
        Optional export dimensions, DPI, and format. None preserves native
        dimensions and the requested path format.

    Returns
    -------
    pandas.DataFrame
        Columns month, plot_date, group, split, observed, fitted, and volume.
        month is month start; plot_date is mean observation time; means use
        weights and zero-weight groups are omitted.

    Raises
    ------
    PlotError
        Vectors are misaligned, timestamps are missing, or numeric values and
        weights are invalid.

    Notes
    -----
    Creates parent directories for save_fig and closes the figure after saving.
    Both samples accept datetime-like values, normalized to UTC before monthly
    grouping. Groups replace project-specific business-type assumptions.
    """
    splits = []
    for name, timestamps, observed, fitted, weight, group in (
        (
            "reference",
            reference_timestamp,
            reference_observed,
            reference_fitted,
            reference_weight,
            reference_group,
        ),
        (
            "current",
            current_timestamp,
            current_observed,
            current_fitted,
            current_weight,
            current_group,
        ),
    ):
        y, prediction = numeric_vector(observed, f"{name}_observed"), numeric_vector(
            fitted, f"{name}_fitted"
        )
        time = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).tz_localize(None)
        same_length(timestamp=time, observed=y, fitted=prediction)
        if time.isna().any():
            raise PlotError("timestamps must not contain missing values")
        exposure = weights(weight, len(y))
        labels = (
            np.full(len(y), "all", dtype=object)
            if group is None
            else np.asarray(group, dtype=object)
        )
        same_length(group=labels, observed=y)
        frame = pd.DataFrame(
            {
                "month": time.to_period("M").to_timestamp(),
                "plot_date": time,
                "group": pd.Series(labels).fillna("missing").astype(str),
                "observed_weighted": y * exposure,
                "fitted_weighted": prediction * exposure,
                "volume": exposure,
            }
        )
        aggregated = frame.groupby(["month", "group"], as_index=False).agg(
            plot_date=("plot_date", "mean"),
            observed_weighted=("observed_weighted", "sum"),
            fitted_weighted=("fitted_weighted", "sum"),
            volume=("volume", "sum"),
        )
        aggregated = aggregated.loc[aggregated.volume > 0].copy()
        aggregated["observed"] = aggregated.observed_weighted / aggregated.volume
        aggregated["fitted"] = aggregated.fitted_weighted / aggregated.volume
        aggregated["split"] = name
        splits.append(
            aggregated[
                ["month", "plot_date", "group", "split", "observed", "fitted", "volume"]
            ]
        )
    grouped = pd.concat(splits, ignore_index=True)
    groups = sorted(grouped.group.unique())
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 5 * len(groups)), sharex=True)
    boundary = pd.Timestamp(
        pd.to_datetime(current_timestamp, utc=True).min()
    ).tz_localize(None)
    for ax, label in zip(np.atleast_1d(axes), groups):
        ax.axvline(boundary, color="tab:orange", linestyle=":", label="Current begins")
        for split, linestyle in (("reference", "-"), ("current", "--")):
            subset = grouped.loc[
                (grouped.group == label) & (grouped.split == split)
            ].sort_values("month")
            if not subset.empty:
                ax.plot(
                    subset.plot_date,
                    subset.observed,
                    color="tab:blue",
                    linestyle=linestyle,
                    marker="o",
                    label=f"Observed — {split}",
                )
                ax.plot(
                    subset.plot_date,
                    subset.fitted,
                    color="tab:red",
                    linestyle=linestyle,
                    marker="o",
                    label=f"Fitted — {split}",
                )
        ax.set(title=label, ylabel="Weighted observed / fitted")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    np.atleast_1d(axes)[-1].set_xlabel("Month")
    fig.suptitle("Monthly observed vs fitted")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, save_fig, config=figure)
    return grouped
