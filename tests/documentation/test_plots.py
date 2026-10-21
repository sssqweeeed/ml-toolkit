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


def test_gallery_check_detects_stale_populated_image(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from examples import render_plots

    expected = tmp_path / "expected"
    generated = tmp_path / "generated"
    expected.mkdir()
    generated.mkdir()
    monkeypatch.setattr(render_plots, "FIGURE_NAMES", ("plot.png",))
    for directory, values in ((expected, [1, 2, 3]), (generated, [3, 2, 1])):
        figure, axis = plt.subplots()
        axis.plot(values)
        figure.savefig(directory / "plot.png")
        plt.close(figure)
    with pytest.raises(AssertionError, match="Stale gallery figure: plot.png"):
        render_plots._compare_gallery_assets(expected, generated)
    (expected / "plot.png").write_bytes((generated / "plot.png").read_bytes())
    render_plots._compare_gallery_assets(expected, generated)
