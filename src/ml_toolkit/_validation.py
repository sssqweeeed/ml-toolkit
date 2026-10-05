import polars as pl


def _validate_object_id(object_id: pl.Series, n_rows: int) -> None:
    if not isinstance(object_id, pl.Series):
        raise TypeError("object_id must be a Polars Series")
    if len(object_id) != n_rows:
        raise ValueError("object_id must have equal lengths with data rows")
    if object_id.dtype != pl.String and not object_id.dtype.is_integer():
        raise TypeError("object_id must have string or integer dtype")
    if object_id.null_count():
        raise ValueError("object_id must not contain null values")
