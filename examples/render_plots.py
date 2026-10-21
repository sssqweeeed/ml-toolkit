"""Regenerate the documented plot gallery and metric example output.

Run ``python -m examples.render_plots`` from a source checkout with the shap
extra installed. ``--check`` renders to a temporary directory and verifies the
captured output and checked-in gallery without modifying them.
"""

from __future__ import annotations

import argparse
import importlib
from contextlib import chdir, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
FIGURE_NAMES = (
    "precision-recall.png",
    "f1-threshold.png",
    "calibration.png",
    "observed-fitted.png",
    "observed-fitted-levels.png",
    "time-observed-fitted.png",
    "gains.png",
    "lift.png",
    "pu-distribution.png",
    "pu-capture.png",
    "psi-scores.png",
    "psi-time.png",
    "time-humidity.png",
    "time-humidity-week.png",
    "feature-importance.png",
    "shap-importance.png",
    "shap-beeswarm.png",
    "shap-temperature.png",
)
EXAMPLE = re.compile(
    r"(?P<example><!-- example: (?P<name>[\w-]+) -->\n+"
    r"```python\n(?P<code>.*?)\n```)"
    r"(?:\n\n<!-- output: (?P=name) -->\n```text\n.*?\n```\n<!-- /output -->)?",
    re.DOTALL,
)


def _execute_examples(
    page: Path,
    namespace: dict[str, object],
    figure_directory: Path | None = None,
) -> str:
    document = page.read_text()
    examples = list(EXAMPLE.finditer(document))
    if len(examples) != len(re.findall(r"^```python$", document, re.MULTILINE)):
        raise ValueError(f"Every Python block in {page.name} needs an example marker")
    names = [example["name"] for example in examples]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate example markers in {page.name}")
    replacements: list[tuple[int, int, str]] = []
    for example in examples:
        captured = StringIO()
        with redirect_stdout(captured):
            exec(
                compile(example["code"], f"{page.name}:{example['name']}", "exec"),
                namespace,
            )
        if example["name"] == "plots-setup":
            namespace["output"] = figure_directory
        replacement = example["example"]
        stdout = "\n".join(
            line.rstrip() for line in captured.getvalue().splitlines()
        ).rstrip("\n")
        if stdout:
            replacement += (
                f"\n\n<!-- output: {example['name']} -->\n```text\n"
                f"{stdout}\n```\n<!-- /output -->"
            )
        replacements.append((example.start(), example.end(), replacement))
    for start, end, replacement in reversed(replacements):
        document = document[:start] + replacement + document[end:]
    return document


def _validate_plot_results(namespace: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import shap

    from examples.gallery_data import BikeGallery, DigitsGallery
    from examples.citation_data import CitationGallery
    from ml_toolkit.metrics import RankingMetrics
    from ml_toolkit.psi import PSIResult

    digits = cast(DigitsGallery, namespace["digits"])
    bikes = cast(BikeGallery, namespace["bikes"])
    assert len(digits.train_indices) + len(digits.test_indices) == 1797
    assert len(digits.observed) == 719
    assert np.intersect1d(digits.train_indices, digits.test_indices).size == 0
    assert np.unique(digits.probabilities).size >= 500
    assert len(bikes.train_indices) + len(bikes.test_indices) == 17379
    assert np.intersect1d(bikes.train_indices, bikes.test_indices).size == 0
    assert len(bikes.reference.observed) >= 4000
    assert len(bikes.current.observed) >= 8000
    assert bikes.reference.time.dt.year().unique().to_list() == [2011]
    assert bikes.current.time.dt.year().unique().to_list() == [2012]

    curve = cast(pd.DataFrame, namespace["curve"])
    assert len(curve) >= 500
    assert curve.equals(cast(pd.DataFrame, namespace["precision_recall_table"]))
    assert curve.equals(cast(pd.DataFrame, namespace["f1_table"]))
    assert curve["threshold"].is_monotonic_decreasing
    for index in np.linspace(0, len(curve) - 1, 6, dtype=int):
        threshold = float(curve["threshold"].iloc[int(index)])
        selected = digits.probabilities >= threshold
        positives = digits.observed[selected].sum()
        np.testing.assert_allclose(
            float(curve["precision"].iloc[int(index)]), positives / selected.sum()
        )
        np.testing.assert_allclose(
            float(curve["recall"].iloc[int(index)]), positives / digits.observed.sum()
        )
    calibration = cast(pd.DataFrame, namespace["calibration_table"])
    assert len(calibration) == 10
    assert calibration["count"].sum() == len(digits.observed)
    assert calibration["count"].between(60, 85).all()

    by_feature = cast(pd.DataFrame, namespace["by_feature"])
    assert 8 <= len(by_feature) <= 10
    assert by_feature["volume"].min() >= 200
    assert by_feature["volume"].sum() == len(bikes.reference.observed)
    np.testing.assert_allclose(
        np.average(by_feature["observed"], weights=by_feature["volume"]),
        bikes.reference.observed.mean(),
    )
    by_level = cast(pd.DataFrame, namespace["by_level"])
    for sample, expected_count in (
        ("reference", len(bikes.reference.observed)),
        ("current", len(bikes.current.observed)),
    ):
        part = by_level.loc[by_level["sample"] == sample]
        assert part["volume"].sum() == expected_count
        assert (part["volume"] > 0).sum() >= 8
    monthly = cast(pd.DataFrame, namespace["temporal"])
    assert len(monthly) == 18
    assert monthly.groupby("split").size().to_dict() == {"current": 12, "reference": 6}
    assert monthly["volume"].min() >= 500
    assert monthly["volume"].sum() == len(bikes.observed)
    assert str(monthly["period_start"].min().date()) == "2011-07-01"
    assert str(monthly["period_start"].max().date()) == "2012-12-01"
    np.testing.assert_allclose(
        np.average(monthly["observed"], weights=monthly["volume"]),
        bikes.observed.mean(),
    )
    distribution = cast(pl.DataFrame, namespace["time_distribution"])
    assert distribution.height == 18
    assert (distribution["p05"] <= distribution["p25"]).all()
    assert (distribution["p25"] <= distribution["p50"]).all()
    assert (distribution["p50"] <= distribution["p75"]).all()
    assert (distribution["p75"] <= distribution["p95"]).all()
    comparison = cast(PSIResult, namespace["comparison"])
    assert comparison.psi.height == bikes.features.width == 11
    assert set(comparison.psi["feature"]) == set(bikes.features.columns)
    assert np.isfinite(comparison.psi["psi"].to_numpy()).all()
    temperature = comparison.time_distribution.filter(
        pl.col("feature") == "temperature"
    )
    shares = temperature.group_by("sample", "period_start").agg(pl.col("share").sum())
    assert shares.height == 18
    np.testing.assert_allclose(shares["share"].to_numpy(), 1.0)
    importance = cast(pd.DataFrame, namespace["importance_table"])
    assert len(importance) == bikes.features.width
    assert set(importance["feature"]) == set(bikes.importance)
    assert importance["importance"].is_monotonic_decreasing
    assert importance["importance"].std() > 0.1

    ranked = cast(RankingMetrics, namespace["ranked"])
    assert len(ranked.cumulative_exposure) == len(bikes.observed) + 1
    assert [point.percent for point in ranked.selected_points] == [10.0, 20.0]
    assert all(point.lift > 1.0 for point in ranked.selected_points)
    np.testing.assert_allclose(ranked.cumulative_observed[[0, -1]], [0.0, 1.0])
    explanation = cast(shap.Explanation, namespace["explanation"])
    contributions = np.asarray(explanation.values, dtype=float)
    assert contributions.shape == (600, bikes.features.width)
    assert (contributions.std(axis=0) > 0.01).sum() >= 5
    assert contributions.std() > 1.0
    base_values = np.asarray(explanation.base_values, dtype=float)
    prediction = bikes.native_model.predict(explanation.data)
    np.testing.assert_allclose(
        contributions.sum(axis=1) + base_values, prediction, atol=1e-6
    )
    citations = cast(CitationGallery, namespace["citations"])
    assert citations.paper_count == 2708 and citations.feature_count == 1433
    assert citations.known_citation_count == 5429
    assert citations.unlabeled_pair_count == 20000
    assert len(citations.labels) == 25429
    assert len(set(citations.pairs)) == len(citations.labels)
    assert citations.labels.sum() == citations.known_citation_count
    assert np.unique(citations.scores).size >= 500
    assert np.std(citations.scores) > 0.01
    assert (
        citations.scores[citations.labels == 1].mean()
        > citations.scores[citations.labels == 0].mean()
    )
    assert not plt.get_fignums(), "A documented plot left a figure open"


def _validate_gallery_assets(directory: Path) -> None:
    import matplotlib.image as mpimg
    import numpy as np

    linked = re.findall(
        r"!\[[^\]]*\]\(\.\./assets/plots/([^)]*)\)",
        (ROOT / "docs/api/plots.md").read_text(),
    )
    assert set(linked) == set(FIGURE_NAMES) and len(linked) == len(FIGURE_NAMES)
    for name in FIGURE_NAMES:
        path = directory / name
        assert path.is_file() and path.stat().st_size > 2000, name
        pixels = mpimg.imread(path)
        assert pixels.shape[0] >= 300 and pixels.shape[1] >= 300, name
        assert np.ptp(pixels[..., :3]) > 0.5, f"Figure has no visible content: {name}"


def _compare_gallery_assets(expected: Path, generated: Path) -> None:
    from matplotlib.testing.compare import compare_images

    for name in FIGURE_NAMES:
        difference = compare_images(
            str(expected / name), str(generated / name), tol=1
        )
        if difference is not None:
            raise AssertionError(
                f"Stale gallery figure: {name}; run python -m examples.render_plots"
            )


def _validate_metric_results(namespace: dict[str, object]) -> None:
    import numpy as np

    from ml_toolkit.metrics import (
        Metric,
        MetricsClassification,
        MetricsFrequency,
        MetricsSeverity,
        MetricScores,
        RankingMetrics,
    )

    np.testing.assert_allclose(
        cast(MetricsClassification, namespace["scores"]).recall, 2 / 3
    )
    np.testing.assert_allclose(
        cast(MetricsFrequency, namespace["frequency"]).prediction_over_observed, 0.86
    )
    np.testing.assert_allclose(
        cast(MetricsSeverity, namespace["severity"]).prediction_over_observed, 6 / 7
    )
    np.testing.assert_allclose(
        cast(RankingMetrics, namespace["ranking"]).normalized_gini, 3 / 7
    )
    assert (
        cast(MetricScores, namespace["pu_scores"]).to_dict()[
            "known_positive_capture_at_10pct"
        ]
        == 0.5
    )
    from ml_toolkit.types import FloatArray

    np.testing.assert_allclose(
        cast(FloatArray, namespace["capacity"]), [0.0, 0.0, 0.5, 0.5, 1.0]
    )
    np.testing.assert_allclose(
        cast(FloatArray, namespace["capture"]), [0.0, 0.5, 0.5, 1.0, 1.0]
    )
    metric = cast(Metric, namespace["metric"])

    np.testing.assert_allclose(
        metric(
            cast(FloatArray, namespace["observed"]),
            cast(FloatArray, namespace["probabilities"]),
            cast(FloatArray, namespace["weight"]),
        ).to_dict()["mae"],
        0.35,
    )


def render(check: bool = False) -> None:
    """Execute the documented recipes and regenerate or verify their outputs.

    Parameters
    ----------
    check : bool, default False
        Render figures in a temporary directory and fail on stale captured
        output or missing checked-in images. False updates both API pages and
        the 18 gallery PNGs in docs/assets/plots.
    """
    try:
        importlib.import_module("shap")
    except ImportError as error:
        raise RuntimeError(
            "Install the shap extra before rendering the complete gallery"
        ) from error
    with (
        TemporaryDirectory(prefix="ml-toolkit-gallery-") as temporary,
        chdir(temporary),
    ):
        os.environ.setdefault("MPLCONFIGDIR", str(Path(temporary) / "matplotlib"))
        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        # SHAP's beeswarm jitter uses NumPy's global random state.
        np.random.seed(42)
        asset_directory = ROOT / "docs/assets/plots"
        figure_directory = Path(temporary) / "plots" if check else asset_directory
        figure_directory.mkdir(parents=True, exist_ok=True)
        namespace: dict[str, object] = {}
        plots = ROOT / "docs/api/plots.md"
        plot_document = _execute_examples(plots, namespace, figure_directory)
        _validate_plot_results(namespace)
        _validate_gallery_assets(figure_directory)
        metrics = ROOT / "docs/api/metrics.md"
        namespace = {}
        metric_document = _execute_examples(metrics, namespace)
        _validate_metric_results(namespace)
        for page, document in ((plots, plot_document), (metrics, metric_document)):
            if check:
                if page.read_text() != document:
                    raise RuntimeError(
                        f"Stale output in {page.name}; run python -m examples.render_plots"
                    )
            else:
                page.write_text(document)
        if check:
            _validate_gallery_assets(asset_directory)
            _compare_gallery_assets(asset_directory, figure_directory)
        print(
            f"Verified {len(EXAMPLE.findall(plot_document))} plot examples, {len(FIGURE_NAMES)} gallery figures, and every metric family."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify without modifying gallery assets or API pages",
    )
    arguments = parser.parse_args()
    render(check=arguments.check)


if __name__ == "__main__":
    main()
