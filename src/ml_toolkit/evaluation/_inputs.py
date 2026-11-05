"""Combine aligned pools for prediction evaluation."""

import polars as pl

from ..data import Pool, validate_auxiliary_pair


def combine_evaluation_pools(train: Pool, heldout: Pool | None) -> Pool:
    if heldout is None:
        return train
    validate_auxiliary_pair(train, heldout)
    if (
        train.X.schema != heldout.X.schema
        or train.cat_features != heldout.cat_features
        or train.numerical_features != heldout.numerical_features
    ):
        raise ValueError("Reporting pools must have matching features and metadata")

    def optional(left: pl.Series | None, right: pl.Series | None) -> pl.Series | None:
        if left is None and right is None:
            return None
        if left is None or right is None:
            raise ValueError("Reporting pools must have matching optional fields")
        return pl.concat([left, right], rechunk=False)

    return Pool(
        X=pl.concat([train.X, heldout.X], rechunk=False),
        y=pl.concat([train.y, heldout.y], rechunk=False),
        time=pl.concat([train.time, heldout.time], rechunk=False),
        weight=pl.concat([train.weight, heldout.weight], rechunk=False),
        auxiliary_features=(
            None
            if train.auxiliary_features is None or heldout.auxiliary_features is None
            else pl.concat(
                [train.auxiliary_features, heldout.auxiliary_features], rechunk=False
            )
        ),
        object_id=optional(train.object_id, heldout.object_id),
        cat_features=train.cat_features,
        numerical_features=train.numerical_features,
    )
