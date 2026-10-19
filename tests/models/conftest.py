from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from examples.project_pool import ProjectPool as Pool


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    n = 48
    return Pool(
        pl.DataFrame(
            {
                "noise": rng.normal(size=n),
                "signal": np.arange(n, dtype=float),
                "category": [10, None, 20] * 16,
            }
        ),
        pl.Series("target", 1.0 + np.arange(n) / 8),
        pl.Series("date", [date(2024, 1, 1) + timedelta(days=i * 5) for i in range(n)]),
        pl.Series("weight", np.linspace(0.5, 2, n)),
        cat_features=("category",),
    )
