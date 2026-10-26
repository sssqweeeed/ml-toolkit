"""Feature conversion shared by pool and prediction preparation."""

from collections.abc import Sequence

import pandas as pd
import polars as pl


def categorical_frame(data: pl.DataFrame, categorical: Sequence[str]) -> pd.DataFrame:
    expressions: list[pl.Expr] = []
    for name in categorical:
        column = pl.col(name)
        dtype = data.schema[name]
        if dtype == pl.Enum:
            column = column.cast(pl.Categorical)
        elif dtype != pl.Categorical:
            column = column.cast(pl.String).cast(pl.Categorical)
        expressions.append(column.fill_null("unknown"))
    # Separate numeric blocks let Arrow share the original numeric buffers.
    return data.with_columns(expressions).to_pandas(split_blocks=True)
