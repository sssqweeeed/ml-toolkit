"""Explicitly download and cache original sources used by documentation recipes.

Run ``python -m examples.prepare_data`` from the checkout before executing the
documentation. This command prepares source data only; it does not fit models.
"""

from __future__ import annotations

import argparse

from examples.citation_data import load_citation_gallery
from examples.gallery_data import load_bike_pool
from examples.retail_data import load_retail_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", choices=("bike", "retail", "cora"),
        help="Prepare only this source; repeat to select several. Default: all.",
    )
    datasets = parser.parse_args().dataset or ("bike", "retail", "cora")
    for dataset in datasets:
        if dataset == "bike":
            load_bike_pool(download_if_missing=True)
        elif dataset == "retail":
            load_retail_pool(download_if_missing=True)
        else:
            load_citation_gallery(download_if_missing=True)
        print(f"Cached {dataset} source data")


if __name__ == "__main__":
    main()
