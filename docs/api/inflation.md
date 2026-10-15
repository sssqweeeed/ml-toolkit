# Inflation

Inflation strategies convert monetary values between price dates. Use a
constant annual rate for a simple assumption or a monthly index when you have
historical or planned index levels.

## Convert to a later price date

```python
from datetime import date

import numpy as np
import polars as pl

from ml_toolkit.inflation import ConstantRateInflation

from_dates = pl.Series([date(2025, 1, 1)])
to_dates = pl.Series([date(2026, 1, 1)])
inflation = ConstantRateInflation(annual_rate=0.08)
adjusted = inflation.adjust(np.array([100.0]), from_dates, to_dates)
print(adjusted.round(2).tolist())  # [107.99]
```

`0.08` means 8% annual compound growth. The result is slightly below `108`
because elapsed years use 365.25 days. Each row can have its own amount and date
pair; reversing the dates converts back to an earlier price level.

## Use a monthly index

```python
from ml_toolkit.inflation import ManualInflation

monthly = ManualInflation(
    {date(2025, 1, 1): 1.0, date(2025, 3, 1): 1.21}
)
adjusted = monthly.adjust(
    np.array([100.0]),
    pl.Series([date(2025, 1, 15)]),
    pl.Series([date(2025, 2, 28)]),
)
print(adjusted.round(2).tolist())  # [110.0]
```

Continuing with the imports above, February's interpolated index is `1.1`.
Every date within a month uses that month's level. Missing months between
anchors are interpolated in log space; requested months must stay within the
anchor range.

To fit and serve an inflation-adjusted regressor, compose a strategy with
`InflationAdjustedModel`; see its constructor in the [model reference](models.md).

## API reference

Import strategies from `ml_toolkit.inflation`.

::: ml_toolkit.inflation
    options:
      members: [InflationAdjuster, ConstantRateInflation, ManualInflation]
