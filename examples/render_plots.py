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
    "monthly-observed-fitted.png",
    "gains.png",
    "lift.png",
    "pu-distribution.png",
    "pu-capture.png",
    "psi-scores.png",
    "psi-monthly.png",
    "monthly-age.png",
    "feature-importance.png",
    "shap-importance.png",
    "shap-beeswarm.png",
    "shap-age.png",
)
EXAMPLE = re.compile(
    r"(?P<example><!-- example: (?P<name>[\w-]+) -->\n"
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
            exec(compile(example["code"], f"{page.name}:{example['name']}", "exec"), namespace)
        if example["name"] == "plots-setup":
            namespace["output"] = figure_directory
        replacement = example["example"]
        stdout = captured.getvalue().rstrip("\n")
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

    from ml_toolkit.metrics import RankingMetrics

    curve = cast(pd.DataFrame, namespace["curve"])
    assert curve.equals(cast(pd.DataFrame, namespace["precision_recall_table"]))
    assert curve.equals(cast(pd.DataFrame, namespace["f1_table"]))
    np.testing.assert_allclose(curve["threshold"], [0.9, 0.7, 0.6, 0.1])
    np.testing.assert_allclose(curve["precision"], [1.0, 1.0, 2 / 3, 0.5])
    assert cast(pd.DataFrame, namespace["calibration_table"])["count"].sum() == 4
    by_feature = cast(pd.DataFrame, namespace["by_feature"])
    np.testing.assert_allclose(by_feature["observed"], [1.5, 3.5])
    np.testing.assert_allclose(by_feature["volume"], [2.0, 2.0])
    by_level = cast(pd.DataFrame, namespace["by_level"])
    unseen = by_level.loc[(by_level["sample"] == "current") & (by_level["level"] == "__UNSEEN__")]
    np.testing.assert_allclose(unseen["observed"], [4.5])
    assert by_level["volume"].sum() == 8
    assert cast(pd.DataFrame, namespace["monthly"])["volume"].sum() == 8
    monthly = cast(pl.DataFrame, namespace["monthly_distribution"])
    np.testing.assert_allclose(monthly["mean"].to_numpy(), [25.0, 45.0, 50.0, 70.0])
    assert cast(pd.DataFrame, namespace["importance_table"])["feature"].tolist() == ["age", "region"]
    ranked = cast(RankingMetrics, namespace["ranked"])
    assert [(point.percent, point.gain, point.lift) for point in ranked.selected_points] == [
        (25.0, 0.5, 2.0)
    ]
    import shap

    explanation = cast(shap.Explanation, namespace["explanation"])
    contributions = np.asarray(explanation.values, dtype=float)
    assert contributions.shape == (24, 2)
    assert np.abs(contributions).max() > 1.0, "SHAP examples must show fitted effects"
    assert not plt.get_fignums(), "A documented plot left a figure open"


def _validate_gallery_assets(directory: Path) -> None:
    import matplotlib.image as mpimg
    import numpy as np

    linked = re.findall(r"!\[[^\]]*\]\(\.\./assets/plots/([^)]*)\)", (ROOT / "docs/api/plots.md").read_text())
    assert set(linked) == set(FIGURE_NAMES) and len(linked) == len(FIGURE_NAMES)
    for name in FIGURE_NAMES:
        path = directory / name
        assert path.is_file() and path.stat().st_size > 2000, name
        pixels = mpimg.imread(path)
        assert pixels.shape[0] >= 300 and pixels.shape[1] >= 300, name
        assert np.ptp(pixels[..., :3]) > 0.5, f"Figure has no visible content: {name}"


def _validate_metric_results(namespace: dict[str, object]) -> None:
    import numpy as np

    from ml_toolkit.metrics import Metric, MetricsClassification, MetricsFrequency, MetricsSeverity, MetricScores, RankingMetrics

    assert cast(MetricsClassification, namespace["scores"]).recall == 1.0
    assert cast(MetricsFrequency, namespace["frequency"]).prediction_over_observed == 1.0
    assert cast(MetricsSeverity, namespace["severity"]).prediction_over_observed == 1.0
    assert cast(RankingMetrics, namespace["ranking"]).normalized_gini == 1.0
    assert cast(MetricScores, namespace["pu_scores"]).to_dict()["known_positive_capture_at_10pct"] == 0.5
    from ml_toolkit.types import FloatArray

    np.testing.assert_allclose(cast(FloatArray, namespace["capacity"]), [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(cast(FloatArray, namespace["capture"]), [0.0, 0.5, 0.5, 1.0, 1.0])
    metric = cast(Metric, namespace["metric"])

    assert metric(
        cast(FloatArray, namespace["observed"]),
        cast(FloatArray, namespace["probabilities"]),
        cast(FloatArray, namespace["weight"]),
    ).to_dict() == {"mae": 0.275}


def render(check: bool = False) -> None:
    """Execute the documented recipes and regenerate or verify their outputs.

    Parameters
    ----------
    check : bool, default False
        Render figures in a temporary directory and fail on stale captured
        output or missing checked-in images. False updates both API pages and
        the 17 gallery PNGs in docs/assets/plots.
    """
    try:
        importlib.import_module("shap")
    except ImportError as error:
        raise RuntimeError("Install the shap extra before rendering the complete gallery") from error
    with TemporaryDirectory(prefix="ml-toolkit-gallery-") as temporary, chdir(temporary):
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
                    raise RuntimeError(f"Stale output in {page.name}; run python -m examples.render_plots")
            else:
                page.write_text(document)
        if check:
            _validate_gallery_assets(asset_directory)
        print("Verified 18 plot examples, 17 gallery figures, and every metric family.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without modifying gallery assets or API pages")
    arguments = parser.parse_args()
    render(check=arguments.check)


if __name__ == "__main__":
    main()
