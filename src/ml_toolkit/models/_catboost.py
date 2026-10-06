"""Small shared operations for CatBoost adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl


def normalize_loss(params):
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


def prepare_frame(data, names, categorical):
    if not isinstance(data, (pl.DataFrame, pd.DataFrame)):
        raise TypeError("Prediction data must be a Polars or Pandas DataFrame")
    missing = sorted(set(names) - set(data.columns))
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    if isinstance(data, pl.DataFrame):
        return (
            data.select(names)
            .with_columns(
                pl.col(name).cast(pl.String).fill_null("unknown")
                for name in categorical
            )
            .to_pandas()
        )
    frame = data.loc[:, names].copy()
    for name in categorical:
        frame[name] = frame[name].astype("string").fillna("unknown").astype(str)
    return frame


def native_fit(model_type, params, train_data, validation_data):
    params = params.copy()
    if validation_data is None:
        params.pop("early_stopping_rounds", None)
        params["use_best_model"] = False
    model = model_type(**params)
    model.fit(train_data, eval_set=validation_data)
    return model


def read_state(context, expected_type, baseline_mode):
    state = json.loads(Path(context.artifacts["config"]).read_text())
    if state.get("version") != 1 or state.get("model_type") != expected_type:
        raise ValueError("Unsupported model state version or model type")
    if state.get("baseline_mode") != baseline_mode:
        raise ValueError("Invalid baseline application mode")
    return state
