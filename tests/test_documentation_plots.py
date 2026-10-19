"""Execute the plotted documentation and verify its published results."""

import pytest


def test_documented_gallery_and_metric_results():
    pytest.importorskip("shap", reason="The complete gallery includes optional SHAP plots")
    from examples.render_plots import render

    render(check=True)
