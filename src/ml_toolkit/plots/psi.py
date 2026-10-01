"""PSI-specific figures; Matplotlib is an optional dependency."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def feature_psi_plot(scores: pl.DataFrame, save_fig: Path) -> None:
    """Save a horizontal bar chart of PSI values by feature."""
    required = {"feature", "psi"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"PSI score columns are missing: {missing}")
    if scores.is_empty():
        raise ValueError("PSI scores must not be empty")
    ordered = scores.select("feature", "psi").sort("psi")
    names = ordered.get_column("feature").to_list()
    values = ordered.get_column("psi").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("PSI scores must be finite")

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(names) + 1)))
    ax.barh(names, values, color="tab:blue")
    ax.set(xlabel="Population Stability Index", ylabel="Feature", title="PSI by feature")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_fig.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(save_fig, bbox_inches="tight")
    finally:
        plt.close(fig)


def monthly_psi_level_distribution(
    distribution: pl.DataFrame,
    feature: str,
    save_fig: Path,
) -> None:
    """Plot precomputed monthly PSI level shares as touching stacked bars."""
    required = {"feature", "month", "sample", "level", "share"}
    missing = sorted(required - set(distribution.columns))
    if missing:
        raise ValueError(f"PSI level distribution columns are missing: {missing}")

    feature_data = distribution.filter(pl.col("feature") == feature)
    if feature_data.is_empty():
        raise ValueError(f"Feature {feature!r} is missing from PSI distribution")

    periods = feature_data.select("sample", "month").unique(maintain_order=True)
    levels = feature_data.get_column("level").unique(maintain_order=True).to_list()
    x = np.arange(len(periods))
    bottom = np.zeros(len(periods), dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.get_cmap("tab20")
    legend_handles = []
    for index, level in enumerate(levels):
        shares = (
            periods.join(
                feature_data.filter(pl.col("level") == level).select(
                    "sample", "month", "share"
                ),
                on=["sample", "month"],
                how="left",
            )
            .get_column("share")
            .fill_null(0.0)
            .to_numpy()
        )
        legend_handles.append(
            ax.bar(
                x,
                shares,
                width=1.0,
                bottom=bottom,
                color=colors(index % 20),
                linewidth=0,
                antialiased=False,
            )
        )
        bottom += shares

    samples = periods.get_column("sample").to_list()
    if "current" in samples:
        current_start = samples.index("current")
        ax.axvline(
            current_start - 0.5,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )
        ax.text(
            -0.45,
            0.98,
            "REFERENCE",
            transform=ax.get_xaxis_transform(),
            va="top",
        )
        ax.text(
            current_start - 0.4,
            0.98,
            "CURRENT",
            transform=ax.get_xaxis_transform(),
            va="top",
        )

    months = periods.get_column("month").to_list()
    tick_positions = np.linspace(0, len(months) - 1, min(6, len(months)), dtype=int)
    ax.set_xticks(
        tick_positions,
        [months[index].strftime("%b %Y") for index in tick_positions],
    )
    ax.set_xlim(-0.5, len(periods) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)])
    ax.set_title(
        f"Monthly PSI level distribution — {feature}",
        loc="left",
        pad=52,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of rows")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        handles=legend_handles,
        labels=levels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=min(6, len(levels)),
        frameon=False,
    )
    fig.tight_layout()
    save_fig.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(save_fig, bbox_inches="tight")
    finally:
        plt.close(fig)
