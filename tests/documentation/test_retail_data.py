"""Monetary examples retain original UCI prices, dates, and invoice groups."""

from datetime import datetime
from hashlib import sha256

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from examples import retail_data

pytestmark = pytest.mark.documentation


@pytest.fixture(scope="module", autouse=True)
def forbid_download():
    def unexpected_download(*args, **kwargs):
        pytest.fail("Documentation tests must not download source data")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(retail_data, "urlretrieve", unexpected_download)
        yield


@pytest.fixture
def workbook(tmp_path):
    pytest.importorskip("openpyxl")
    # 1010, 1017 and 1055 satisfy the published hash rule; 1000 does not.
    frame = pd.DataFrame(
        [
            ("1010", "A", 2, datetime(2011, 8, 15, 9, 30), 3.5, "France"),
            ("1010", "B", 1, datetime(2011, 8, 15, 9, 30), 1.25, "France"),
            ("1017", "C", 3, datetime(2011, 9, 5, 14), 4.75, "Germany"),
            ("1055", "D", 1, datetime(2011, 10, 2, 11), 8.0, "UK"),
            ("1000", "E", 2, datetime(2011, 8, 16), 2.0, "UK"),
            ("C1004", "A", 2, datetime(2011, 8, 17), 3.5, "France"),
            ("1010", "F", 0, datetime(2011, 8, 15), 3.0, "France"),
            ("1010", "G", -1, datetime(2011, 8, 15), 3.0, "France"),
            ("1010", "H", 1, datetime(2011, 8, 15), 0.0, "France"),
            ("1010", "I", 1, datetime(2011, 8, 15), -1.0, "France"),
            ("1055", "J", 1, datetime(2011, 12, 1), 8.0, "UK"),
        ],
        columns=retail_data.SOURCE_COLUMNS,
    )
    frame["CustomerID"] = 12345
    frame["Description"] = "Unused private source field"
    path = tmp_path / retail_data.WORKBOOK
    frame.to_excel(path, index=False)
    return tmp_path, frame


def test_missing_cache_never_downloads_or_creates_directories(tmp_path):
    directory = tmp_path / "absent"
    assert not retail_data.retail_cache_available(directory)
    for loader in (
        retail_data.load_retail_records,
        retail_data.load_retail_pool,
        retail_data.load_retail_splits,
    ):
        with pytest.raises(FileNotFoundError, match="Online Retail data is not cached"):
            loader(data_home=directory, download_if_missing=False)
    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


def test_workbook_filter_keeps_all_eligible_invoice_lines_and_original_rows(workbook):
    directory, source = workbook
    path = directory / retail_data.WORKBOOK
    before = path.stat()
    records = retail_data.load_retail_records(data_home=directory)
    repeated = retail_data.load_retail_records(data_home=directory)
    assert_frame_equal(records, repeated)
    assert records["source_row"].to_list() == [0, 1, 2, 3]
    assert records["InvoiceNo"].to_list() == ["1010", "1010", "1017", "1055"]
    assert records["UnitPrice"].to_list() == source["UnitPrice"][:4].tolist()
    assert records["InvoiceDate"].to_list() == source["InvoiceDate"][:4].tolist()
    assert set(records.columns) == {"source_row", *retail_data.SOURCE_COLUMNS}
    assert list(directory.iterdir()) == [path]
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


@pytest.mark.parametrize("boundary", [datetime(2011, 9, 1), datetime(2011, 10, 1)])
def test_invoice_crossing_a_documented_split_is_rejected(workbook, boundary):
    directory, source = workbook
    source.loc[1, "InvoiceDate"] = boundary
    source.to_excel(directory / retail_data.WORKBOOK, index=False)
    with pytest.raises(ValueError, match="invoice crosses"):
        retail_data.load_retail_records(data_home=directory)


@pytest.mark.parametrize("column,value,message", [
    ("Country", None, "null"),
    ("UnitPrice", float("inf"), "finite"),
    ("Quantity", float("nan"), "finite"),
])
def test_invalid_required_source_values_are_rejected(tmp_path, column, value, message):
    records = pl.DataFrame({
        "source_row": [0], "InvoiceNo": ["1010"], "StockCode": ["A"],
        "Quantity": [1.0], "InvoiceDate": [datetime(2011, 8, 15)],
        "UnitPrice": [3.5], "Country": ["France"],
    }).with_columns(pl.lit(value).alias(column))
    records.write_parquet(tmp_path / "records.parquet")
    with pytest.raises(ValueError, match=message):
        retail_data.load_retail_records(data_home=tmp_path)


@pytest.fixture(scope="module")
def original_source():
    path = retail_data.DEFAULT_RETAIL_CACHE / retail_data.WORKBOOK
    if not path.is_file():
        pytest.skip("Cache the original UCI Online Retail workbook before source checks")
    pytest.importorskip("openpyxl")
    # Read the original workbook once, independently of any normalized cache.
    frame = pd.read_excel(
        path, engine="openpyxl", usecols=list(retail_data.SOURCE_COLUMNS),
        dtype={"InvoiceNo": str, "StockCode": str, "Country": str},
    )
    return pl.from_pandas(frame).with_row_index("source_row")


@pytest.fixture(scope="module")
def real_records(original_source):
    return retail_data.load_retail_records(download_if_missing=False)


def test_real_sample_preserves_original_prices_dates_and_every_selected_line(
    original_source, real_records,
):
    source = original_source
    original_rows = source[real_records["source_row"].to_list()]
    assert_frame_equal(real_records, original_rows)
    eligible = source.filter(
        ~pl.col("InvoiceNo").str.to_uppercase().str.starts_with("C"),
        pl.col("Quantity") > 0,
        pl.col("UnitPrice") > 0,
        pl.col("InvoiceDate") < datetime(2011, 12, 1),
    )
    selected_invoices = {
        invoice for invoice in eligible["InvoiceNo"].unique()
        if int(sha256(invoice.encode()).hexdigest()[:16], 16) % 10 == 0
    }
    assert set(real_records["InvoiceNo"]) == selected_invoices
    expected_rows = eligible.filter(pl.col("InvoiceNo").is_in(selected_invoices))
    assert set(real_records["source_row"]) == set(expected_rows["source_row"])
    assert_frame_equal(real_records, real_records.sort("InvoiceDate", "source_row"))


def test_real_pool_uses_original_target_without_target_leakage(real_records, monkeypatch):
    monkeypatch.setattr(retail_data, "load_retail_records", lambda **kwargs: real_records)
    pool = retail_data.load_retail_pool()
    assert_frame_equal(pool.X, real_records.select(
        pl.col("StockCode").alias("stock_code"),
        pl.col("Country").alias("country"),
        pl.col("Quantity").alias("quantity"),
    ))
    assert_series_equal(pool.y, real_records["UnitPrice"].cast(pl.Float64))
    assert_series_equal(pool.time, real_records["InvoiceDate"])
    assert_series_equal(pool.object_id, real_records["source_row"])
    np.testing.assert_array_equal(pool.weight, np.ones(len(real_records)))
    assert pool.cat_features == ("stock_code", "country")
    assert pool.numerical_features == ("quantity",)


def test_real_calendar_splits_keep_invoices_disjoint_and_cover_the_sample(
    real_records, monkeypatch,
):
    monkeypatch.setattr(retail_data, "load_retail_records", lambda **kwargs: real_records)
    splits = retail_data.load_retail_splits()
    pools = (splits.train, splits.validation, splits.test)
    assert all(len(pool) > 0 for pool in pools)
    assert splits.train.time.max() < datetime(2011, 9, 1)
    assert splits.validation.time.min() >= datetime(2011, 9, 1)
    assert splits.validation.time.max() < datetime(2011, 10, 1)
    assert splits.test.time.min() >= datetime(2011, 10, 1)
    assert splits.test.time.max() < datetime(2011, 12, 1)
    rows = [set(pool.object_id) for pool in pools]
    assert rows[0].isdisjoint(rows[1] | rows[2])
    assert rows[1].isdisjoint(rows[2])
    assert set.union(*rows) == set(real_records["source_row"])
    invoices = [set(real_records.filter(pl.col("source_row").is_in(group))["InvoiceNo"])
                for group in rows]
    assert invoices[0].isdisjoint(invoices[1] | invoices[2])
    assert invoices[1].isdisjoint(invoices[2])


def test_normalized_cache_reads_reproducibly_without_workbook_or_writes(workbook):
    directory, source = workbook
    expected = retail_data.load_retail_records(data_home=directory)
    path = directory / "records.parquet"
    pl.from_pandas(source[list(retail_data.SOURCE_COLUMNS)]).with_row_index(
        "source_row"
    ).write_parquet(path)
    (directory / retail_data.WORKBOOK).unlink()
    before = path.stat()
    assert retail_data.retail_cache_available(directory)
    for _ in range(2):
        assert_frame_equal(retail_data.load_retail_records(data_home=directory), expected)
    assert list(directory.iterdir()) == [path]
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
