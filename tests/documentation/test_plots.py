"""Execute real-data plotting recipes without implicitly downloading datasets."""

import pytest

pytestmark = pytest.mark.documentation


def test_documented_gallery_and_metric_results():
    from examples.gallery_data import bike_cache_available
    from examples.citation_data import citation_cache_available

    if not bike_cache_available() or not citation_cache_available():
        pytest.skip("Run python -m examples.render_plots once to cache UCI Bike Sharing and Cora data")
    pytest.importorskip("shap", reason="The complete gallery includes optional SHAP plots")
    from examples.render_plots import render

    render(check=True)
