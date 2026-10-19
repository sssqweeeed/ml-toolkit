import numpy as np

from ml_toolkit.metrics.ranking import ranking_metrics as ranking_curve
from ml_toolkit.plots.ranking import gains_curve, lift_curve


def test_ranking_curve_and_figures(tmp_path):
    curve = ranking_curve(
        [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], [1, 1, 1, 1], selected_percents=(50,)
    )
    assert curve.selected_points[0].gain == 1
    assert curve.selected_points[0].lift == 2
    assert np.isclose(curve.normalized_gini, 1)
    gains_curve(curve, tmp_path / "gains.png")
    lift_curve(curve, tmp_path / "lift.png")
    assert (tmp_path / "gains.png").is_file() and (tmp_path / "lift.png").is_file()
