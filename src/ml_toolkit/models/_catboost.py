"""Small shared operations for CatBoost adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar, cast

import catboost
import numpy as np
import pandas as pd
import polars as pl

from .._model_input import categorical_frame
from ..types import ModelParams

NativeT = TypeVar("NativeT", bound=catboost.CatBoost)


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
    for name in categorical:
        column = frame[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            labels = column.cat.categories.astype("string")
            if not column.cat.categories.equals(labels):
                # Convert dictionary entries, never a string for every row.
                codes, categories = pd.factorize(labels, sort=False)
                row_codes = cast(pd.Categorical, column.array).codes
                mapping = np.append(codes, -1).astype(row_codes.dtype, copy=False)
                mapped = mapping[row_codes]
                column = pd.Series(
                    pd.Categorical.from_codes(mapped, categories=categories),
                    index=column.index,
                    name=name,
                )
            if column.isna().any():
                if "unknown" not in column.cat.categories:
                    column = column.cat.add_categories(["unknown"])
                column = column.fillna("unknown")
            frame[name] = column
        else:
            frame[name] = column.astype("string").fillna("unknown").astype("category")
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


def read_state(
    context: Any, expected_type: str, baseline_mode: str
) -> dict[str, object]:
    state = cast(
        dict[str, object], json.loads(Path(context.artifacts["config"]).read_text())
    )
    if state.get("version") != 1 or state.get("model_type") != expected_type:
        raise ValueError("Unsupported model state version or model type")
    if state.get("baseline_mode") != baseline_mode:
        raise ValueError("Invalid baseline application mode")
    return state


def refit_parameters(params: ModelParams, best_iteration: int | None) -> ModelParams:
    result = params.copy()
    if best_iteration is not None:
        aliases = ("iterations", "n_estimators", "num_trees", "num_boost_round")
        key = next((name for name in aliases if name in result), "iterations")
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
