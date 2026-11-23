"""Small shared operations for CatBoost adapters."""

from __future__ import annotations

from typing import TypeVar

import catboost
import numpy as np
import pandas as pd
import polars as pl

from .._model_input import categorical_frame
from ..types import ModelParams

NativeT = TypeVar("NativeT", bound=catboost.CatBoost)

ITERATION_ALIASES = ("iterations", "n_estimators", "num_trees", "num_boost_round")


def normalize_loss(params: ModelParams) -> ModelParams:
    params = params.copy()
    if "loss" in params:
        loss = params.pop("loss")
        native = params.get("loss_function")
        same_loss = native is loss or (
            isinstance(native, str) and isinstance(loss, str) and native == loss
        )
        if native is not None and not same_loss:
            raise ValueError("Conflicting loss and loss_function values")
        params["loss_function"] = loss
    return params


def prepare_frame(
    data: pl.DataFrame | pd.DataFrame, names: list[str], categorical: list[str]
) -> pd.DataFrame:
    missing = sorted(set(names) - set(data.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    if isinstance(data, pl.DataFrame):
        return categorical_frame(data.select(names), categorical)
    # Construct a new column container without copying numeric buffers, also
    # with pandas versions where selecting columns via .loc makes a deep copy.
    frame = pd.DataFrame(
        {name: data[name] for name in names}, index=data.index, copy=False
    )
    if categorical:
        # Reuse the training conversion so category strings match exactly.
        columns = frame[list(categorical)]
        # Mixed-type object values (also as pandas Categorical dictionaries) cannot
        # convert to Arrow; format them as pandas strings first. They cannot
        # originate from the Polars training path.
        mixed = [
            name
            for name in categorical
            if columns[name].dtype == object
            or (
                isinstance(columns[name].dtype, pd.CategoricalDtype)
                and columns[name].cat.categories.dtype == object
            )
        ]
        columns = columns.astype({name: "string" for name in mixed})
        converted = categorical_frame(pl.from_pandas(columns), categorical)
        converted.index = frame.index
        for name in categorical:
            frame[name] = converted[name]
    return frame


def native_fit(
    model_type: type[NativeT],
    params: ModelParams,
    train_data: catboost.Pool,
    validation_data: catboost.Pool | None,
) -> NativeT:
    params = params.copy()
    if validation_data is None:
        params.pop("early_stopping_rounds", None)
        params["use_best_model"] = False
    model = model_type(**params)
    model.fit(train_data, eval_set=validation_data)
    return model


def refit_parameters(params: ModelParams, best_iteration: int | None) -> ModelParams:
    result = params.copy()
    if best_iteration is not None:
        key = next((name for name in ITERATION_ALIASES if name in result), "iterations")
        result[key] = int(best_iteration) + 1
    for key in ("early_stopping_rounds", "od_type", "od_wait", "od_pval"):
        result.pop(key, None)
    result["use_best_model"] = False
    return result


def feature_importances(
    native: catboost.CatBoost, columns: list[str]
) -> dict[str, float]:
    names = list(native.feature_names_)
    values = np.asarray(
        native.get_feature_importance(type="PredictionValuesChange"), dtype=float
    )
    if not np.isfinite(values).all():
        counts = native.get_tree_leaf_counts()
        leaves = native.get_leaf_values()
        if len(leaves) and leaves.size % int(np.sum(counts)) == 0:
            outputs = leaves.reshape(int(np.sum(counts)), -1)
            start = 0
            constant = True
            for count in counts:
                tree = outputs[start : start + count]
                constant = constant and bool((tree == tree[0]).all())
                start += count
            if constant:
                values = np.zeros(len(names))
    if names != columns or values.shape != (len(names),):
        raise ValueError(
            "Native feature names and dimensions must match the input columns"
        )
    if not np.isfinite(values).all():
        raise ValueError("Native importances must be finite")
    return dict(zip(names, values.tolist(), strict=True))
