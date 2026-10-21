"""Original UCI Online Retail transactions for monetary documentation examples.

Source: https://archive.ics.uci.edu/dataset/352/online+retail (CC BY 4.0).
Prices are sterling per item; InvoiceDate is the original transaction timestamp.
The deterministic invoice sample retains every eligible line of selected invoices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import polars as pl

from ml_toolkit.data import Pool

RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DEFAULT_RETAIL_CACHE = Path(__file__).resolve().parents[1] / ".cache/uci-online-retail"
WORKBOOK = "Online Retail.xlsx"
SOURCE_COLUMNS = ("InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "Country")


@dataclass(frozen=True)
class RetailSplits:
    """Chronological pools of original monetary observations.

    Attributes
    ----------
    train, validation, test : Pool
        Before September 2011, September 2011, and October–November 2011.
        Targets are sterling per item; weights give each invoice line equal weight.
    """

    train: Pool
    validation: Pool
    test: Pool


def retail_cache_available(data_home: Path | None = None) -> bool:
    """Check source availability without downloading or writing files.

    Parameters
    ----------
    data_home : pathlib.Path or None, default None
        Cache directory; None uses the checkout's .cache/uci-online-retail.

    Returns
    -------
    bool
        Whether the original workbook or its normalized cache is available.
    """
    directory = DEFAULT_RETAIL_CACHE if data_home is None else data_home
    return (directory / WORKBOOK).is_file() or (directory / "records.parquet").is_file()


def load_retail_records(
    *, data_home: Path | None = None, download_if_missing: bool = False
) -> pl.DataFrame:
    """Read a reproducible invoice sample without altering prices or timestamps.

    Parameters
    ----------
    data_home : pathlib.Path or None, default None
        Cache directory; None uses the checkout's .cache/uci-online-retail.
    download_if_missing : bool, default False
        Permit downloading the official archive and caching normalized records.
        Reading the workbook requires the example-only openpyxl dependency.

    Returns
    -------
    polars.DataFrame
        Original source columns plus zero-based source_row, sorted by InvoiceDate
        and source_row. Cancellations, nonpositive quantities/prices, and records
        from December 2011 onward are excluded. An invoice is selected when the
        first eight SHA-256 digest bytes of its UTF-8 ID, interpreted big-endian,
        modulo ten equal zero: approximately 10% of invoices, not 10% of lines.
        Customer identifiers and descriptions are not loaded.

    Raises
    ------
    FileNotFoundError
        No cached source exists and downloading is disabled. No files are written
        and no network requests occur in this case.
    ValueError
        Required fields are null, numeric fields nonfinite, or an invoice crosses
        a documented train/validation/test boundary.
    """
    directory = DEFAULT_RETAIL_CACHE if data_home is None else data_home
    workbook = directory / WORKBOOK
    normalized = directory / "records.parquet"
    if normalized.is_file():
        records = pl.read_parquet(normalized)
    else:
        if not workbook.is_file():
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Online Retail data is not cached at {directory}; "
                    "run python -m examples.prepare_data first"
                )
            directory.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="retail-download-", dir=directory) as temporary:
                archive_path = Path(temporary) / "retail.zip"
                urlretrieve(RETAIL_URL, archive_path)
                with ZipFile(archive_path) as archive:
                    extracted = Path(temporary) / WORKBOOK
                    extracted.write_bytes(archive.read(WORKBOOK))
                    extracted.replace(workbook)
        records = pl.from_pandas(pd.read_excel(
            workbook, engine="openpyxl", usecols=list(SOURCE_COLUMNS),
            dtype={"InvoiceNo": str, "StockCode": str, "Country": str},
        )).with_row_index("source_row")
        if download_if_missing:
            records.write_parquet(normalized)

    if any(records[name].null_count() for name in SOURCE_COLUMNS):
        raise ValueError("Required Online Retail fields contain null values")
    if not records.select(
        pl.col("Quantity").is_finite().all() & pl.col("UnitPrice").is_finite().all()
    ).item():
        raise ValueError("Online Retail quantity and price must be finite")
    eligible = records.filter(
        ~pl.col("InvoiceNo").str.to_uppercase().str.starts_with("C"),
        pl.col("Quantity") > 0,
        pl.col("UnitPrice") > 0,
        pl.col("InvoiceDate") < datetime(2011, 12, 1),
    )
    selected = [
        invoice for invoice in eligible["InvoiceNo"].unique().to_list()
        if int.from_bytes(sha256(invoice.encode("utf-8")).digest()[:8], "big") % 10 == 0
    ]
    sampled = eligible.filter(pl.col("InvoiceNo").is_in(selected)).sort(
        "InvoiceDate", "source_row"
    )
    for boundary in (datetime(2011, 9, 1), datetime(2011, 10, 1)):
        spans = sampled.group_by("InvoiceNo").agg(
            pl.col("InvoiceDate").min().alias("first"),
            pl.col("InvoiceDate").max().alias("last"),
        )
        if spans.filter((pl.col("first") < boundary) & (pl.col("last") >= boundary)).height:
            raise ValueError("An invoice crosses a documented temporal boundary")
    return sampled


def load_retail_pool(
    *, data_home: Path | None = None, download_if_missing: bool = False
) -> Pool:
    """Prepare sampled positive item prices with original transaction dates.

    Parameters
    ----------
    data_home : pathlib.Path or None, default None
        Source cache passed to load_retail_records.
    download_if_missing : bool, default False
        Explicitly permit the source download and normalized cache creation.

    Returns
    -------
    Pool
        Features stock_code and country are categorical; quantity is numerical.
        The target is UnitPrice in sterling, time is InvoiceDate, object_id is
        the original source row, and each invoice line has weight one.
        Invoice IDs, customer IDs, dates, and target-derived amounts are excluded
        from model features. Sampling and filtering follow load_retail_records.
    """
    records = load_retail_records(data_home=data_home, download_if_missing=download_if_missing)
    return Pool(
        X=records.select("StockCode", "Country", "Quantity").rename({
            "StockCode": "stock_code", "Country": "country", "Quantity": "quantity",
        }),
        y=records["UnitPrice"].cast(pl.Float64),
        time=records["InvoiceDate"],
        weight=pl.Series("weight", np.ones(len(records), dtype=np.float64)),
        cat_features=("stock_code", "country"),
        numerical_features=("quantity",),
        object_id=records["source_row"],
    )


def load_retail_splits(
    *, data_home: Path | None = None, download_if_missing: bool = False
) -> RetailSplits:
    """Return training, validation and test pools separated by calendar months.

    Parameters
    ----------
    data_home : pathlib.Path or None, default None
        Cache passed to load_retail_pool.
    download_if_missing : bool, default False
        Explicitly permit downloading the source and caching its normalized form.

    Returns
    -------
    RetailSplits
        Training ends before September 2011; validation is September; testing
        is October–November. Eligible lines of an invoice stay in one partition.
    """
    data = load_retail_pool(data_home=data_home, download_if_missing=download_if_missing)
    return RetailSplits(
        train=data.take(np.flatnonzero((data.time < datetime(2011, 9, 1)).to_numpy())),
        validation=data.take(np.flatnonzero((
            (data.time >= datetime(2011, 9, 1)) & (data.time < datetime(2011, 10, 1))
        ).to_numpy())),
        test=data.take(np.flatnonzero((data.time >= datetime(2011, 10, 1)).to_numpy())),
    )
