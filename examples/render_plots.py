"""Regenerate the evaluation guides' captured output and real-data figures.

Run ``python -m examples.render_plots`` after explicitly preparing the source
caches. ``--check`` renders into temporary directories and detects stale outputs
and figures without modifying the documented files.
"""

from __future__ import annotations

import argparse
from contextlib import chdir, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "docs/evaluation/index.md",
    "docs/evaluation/model.md",
    "docs/evaluation/permutation.md",
    "docs/evaluation/robustness.md",
    "docs/evaluation/metric-trend.md",
    "docs/evaluation/classification.md",
    "docs/evaluation/observed-fitted.md",
    "docs/evaluation/ranking.md",
    "docs/evaluation/pu.md",
    "docs/data-analysis/distributions.md",
    "docs/data-analysis/index.md",
    "docs/data-analysis/custom.md",
    "docs/api/psi.md",
    "docs/evaluation/explainability.md",
    "docs/evaluation/custom.md",
)
FIGURE_NAMES = (
    "shrink-mae.png",
    "permutation-mae.png",
    "robustness-missing.png",
    "metric-trend-mae.png",
    "precision_recall/test/precision_recall.png",
    "f1_threshold/test/f1_threshold.png",
    "calibration/test/calibration.png",
    "observed_fitted/000_temperature.png",
    "observed-fitted-levels.png",
    "time_observed_fitted/time.png",
    "gains/test/gains.png",
    "lift/test/lift.png",
    "pu-distribution.png",
    "pu-capture.png",
    "psi-scores.png",
    "psi-time.png",
    "numerical_feature_distributions/000_humidity.png",
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
    page: Path, namespace: dict[str, object], figure_directory: Path
) -> str:
    document = page.read_text()
    examples = list(EXAMPLE.finditer(document))
    blocks = re.findall(r"^```python\n(.*?)\n```", document, re.MULTILINE | re.DOTALL)
    if len(examples) != sum("--8<--" not in code for code in blocks):
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
        if example["name"].endswith("-setup"):
            namespace["output"] = figure_directory
        replacement = example["example"]
        stdout = "\n".join(
            line.rstrip() for line in captured.getvalue().splitlines()
        ).rstrip("\n")
        if stdout:
            replacement += f"\n\n<!-- output: {example['name']} -->\n```text\n{stdout}\n```\n<!-- /output -->"
        replacements.append((example.start(), example.end(), replacement))
    for start, end, replacement in reversed(replacements):
        document = document[:start] + replacement + document[end:]
    return document.rstrip() + "\n"


def _validate_results(page: str, namespace: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import shap

    from examples.evaluation_data import BikeEvaluation
    from examples.citation_data import CitationGallery
    from ml_toolkit.metrics import RankingMetrics
    from ml_toolkit.psi import PSIResult

    if "example" in namespace:
        example = cast(BikeEvaluation, namespace["example"])
        assert len(example.data) == 17379
        assert len(example.train) > 8000 and len(example.test) > 8000
        assert example.train.time.dt.year().unique().to_list() == [2011]
        assert example.test.time.dt.year().unique().to_list() == [2012]
        np.testing.assert_allclose(
            example.prediction[len(example.train) :],
            example.model.predict_pool(example.test),
        )
        assert example.split.to_list() == ["train"] * len(example.train) + [
            "test"
        ] * len(example.test)
    if page.endswith("metric-trend.md"):
        trend = cast(pl.DataFrame, namespace["table"])
        assert trend["period_start"].n_unique() == 24
        assert trend["metric"].unique().to_list() == ["mae"]
        assert trend["sum_weight"].min() > 0
    elif page.endswith("robustness.md"):
        table = cast(pl.DataFrame, namespace["table"])
        assert table.height == 31
        baseline = table.filter(pl.col("is_baseline"))["value"][0]
        unmodified = table.filter((~pl.col("is_baseline")) & (pl.col("level") == 0))
        np.testing.assert_allclose(unmodified["value"].to_numpy(), baseline)
    elif page.endswith("permutation.md"):
        importance = cast(pl.DataFrame, namespace["importance"])
        assert importance.height == 12
        assert importance["features"].list.len().max() == 2
        assert importance["baseline"].n_unique() == 1
        assert importance["importance"].is_finite().all()
    elif page.endswith("model.md"):
        table = cast(pl.DataFrame, namespace["table"])
        assert table.height == 24
        assert table["n_iterations"].n_unique() == 12
        for part, pool in (("train", example.train), ("test", example.test)):
            prediction = example.model.predict_pool(pool)
            expected = np.average(
                abs(pool.y.to_numpy() - prediction), weights=pool.weight.to_numpy()
            )
            actual = table.filter(
                (pl.col("split") == part)
                & (pl.col("n_iterations") == example.model.n_iterations)
            )["value"][0]
            np.testing.assert_allclose(actual, expected)
    elif page.endswith("classification.md"):
        curve = cast(pd.DataFrame, namespace["curve"])
        assert len(curve) > 1000
        assert curve.equals(cast(pd.DataFrame, namespace["precision_recall_table"]))
        assert curve.equals(cast(pd.DataFrame, namespace["f1_table"]))
        calibration = cast(pd.DataFrame, namespace["calibration_table"])
        assert calibration["count"].sum() == len(example.test)
        assert calibration["weight_sum"].sum() == example.test.weight.sum()
        for index in np.linspace(0, len(curve) - 1, 6, dtype=int):
            chosen = (
                example.prediction[len(example.train) :]
                >= curve["threshold"].iloc[int(index)]
            )
            labels = example.test.y.to_numpy()
            np.testing.assert_allclose(
                curve["precision"].iloc[int(index)], labels[chosen].mean()
            )
            np.testing.assert_allclose(
                curve["recall"].iloc[int(index)], labels[chosen].sum() / labels.sum()
            )
    elif page.endswith("observed-fitted.md"):
        by_level = cast(pd.DataFrame, namespace["by_level"])
        for sample, pool in (("reference", example.train), ("current", example.test)):
            part = by_level.loc[by_level["sample"] == sample]
            assert part["volume"].sum() == len(pool)
            populated = part.loc[part["volume"] > 0]
            np.testing.assert_allclose(
                np.average(populated["observed"], weights=populated["volume"]),
                pool.y.mean(),
            )
        temporal = cast(pd.DataFrame, namespace["temporal"])
        assert len(temporal) == 24
        assert temporal.groupby("split").size().to_dict() == {
            "current": 12,
            "reference": 12,
        }
        assert temporal["volume"].min() > 500
        assert temporal["volume"].sum() == len(example.data)
    elif page.endswith("ranking.md"):
        ranked = cast(RankingMetrics, namespace["ranked"])
        assert len(ranked.cumulative_exposure) == len(example.test) + 1
        assert ranked.selected_points[0].percent == 10
        assert ranked.selected_points[0].lift > 1
        np.testing.assert_allclose(ranked.cumulative_observed[[0, -1]], [0, 1])
    elif page.endswith("pu.md"):
        citations = cast(CitationGallery, namespace["citations"])
        assert citations.known_citation_count == 5429
        assert citations.unlabeled_pair_count == 20000
        assert len(set(citations.pairs)) == len(citations.labels)
        assert np.unique(citations.scores).size > 500
        assert (
            citations.scores[citations.labels == 1].mean()
            > citations.scores[citations.labels == 0].mean()
        )
    elif page.endswith("distributions.md"):
        distribution = cast(pl.DataFrame, namespace["time_distribution"])
        assert distribution.height == 24
        for low, high in (
            ("p05", "p25"),
            ("p25", "p50"),
            ("p50", "p75"),
            ("p75", "p95"),
        ):
            assert (distribution[low] <= distribution[high]).all()
    elif page.endswith("psi.md"):
        comparison = cast(PSIResult, namespace["weather_comparison"])
        assert comparison.psi.height == 11
        assert np.isfinite(comparison.psi["psi"].to_numpy()).all()
        shares = (
            comparison.time_distribution.filter(pl.col("feature") == "temperature")
            .group_by("sample", "period_start")
            .agg(pl.col("share").sum())
        )
        assert shares.height == 24
        np.testing.assert_allclose(shares["share"].to_numpy(), 1)
    elif page.endswith("explainability.md"):
        explanation = cast(shap.Explanation, namespace["explanation"])
        contributions = np.asarray(explanation.values, dtype=float)
        assert contributions.shape == (600, example.data.X.width)
        assert (contributions.std(axis=0) > 0.01).sum() >= 5
        np.testing.assert_allclose(
            contributions.sum(axis=1) + explanation.base_values,
            example.model.native_model.predict(explanation.data),
            atol=1e-6,
        )
    assert not plt.get_fignums(), f"{page} left a figure open"


def _validate_gallery_assets(directory: Path) -> None:
    import matplotlib.image as mpimg
    import numpy as np

    linked = [
        name
        for page in PAGES
        for name in re.findall(
            r"!\[[^\]]*\]\(\.\./assets/plots/([^)]*)\)", (ROOT / page).read_text()
        )
    ]
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
        if (
            compare_images(str(expected / name), str(generated / name), tol=1)
            is not None
        ):
            raise AssertionError(
                f"Stale gallery figure: {name}; run python -m examples.render_plots"
            )


def _compare_evaluator_figures(directory: Path) -> None:
    from matplotlib.testing.compare import compare_images

    pairs = (
        ("precision_recall/test/precision_recall.png", "direct-precision-recall.png"),
        ("f1_threshold/test/f1_threshold.png", "direct-f1-threshold.png"),
        ("calibration/test/calibration.png", "direct-calibration.png"),
        ("observed_fitted/000_temperature.png", "direct-observed-fitted.png"),
        ("time_observed_fitted/time.png", "direct-time-observed-fitted.png"),
        ("gains/test/gains.png", "direct-gains.png"),
        ("lift/test/lift.png", "direct-lift.png"),
        ("numerical_feature_distributions/000_humidity.png", "direct-humidity.png"),
    )
    for evaluator, direct in pairs:
        assert (
            compare_images(str(directory / evaluator), str(directory / direct), tol=0)
            is None
        ), evaluator


def render(check: bool = False) -> None:
    """Execute each guide independently, then verify or refresh its outputs."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    with TemporaryDirectory(prefix="ml-toolkit-gallery-") as temporary:
        temporary_root = Path(temporary)
        os.environ.setdefault("MPLCONFIGDIR", str(temporary_root / "matplotlib"))
        figure_directory = temporary_root / "plots"
        figure_directory.mkdir()
        documents: dict[Path, str] = {}
        for index, relative in enumerate(PAGES):
            page_directory = temporary_root / f"page-{index}"
            page_directory.mkdir()
            np.random.seed(42)
            with chdir(page_directory):
                namespace: dict[str, object] = {"__name__": "__documentation__"}
                page = ROOT / relative
                documents[page] = _execute_examples(page, namespace, figure_directory)
                _validate_results(relative, namespace)
            print(f"PASS {relative}", flush=True)
        _validate_gallery_assets(figure_directory)
        _compare_evaluator_figures(figure_directory)
        asset_directory = ROOT / "docs/assets/plots"
        if check:
            for page, document in documents.items():
                if page.read_text() != document:
                    raise AssertionError(
                        f"Stale output in {page}; run python -m examples.render_plots"
                    )
            _validate_gallery_assets(asset_directory)
            _compare_gallery_assets(asset_directory, figure_directory)
        else:
            for page, document in documents.items():
                page.write_text(document)
            for name in FIGURE_NAMES:
                target = asset_directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(figure_directory / name, target)
        print(
            f"Verified {len(PAGES)} guides and {len(FIGURE_NAMES)} figures.", flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify without modifying documented output",
    )
    render(check=parser.parse_args().check)


if __name__ == "__main__":
    main()
