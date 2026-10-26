"""Typed, JSON-safe encoding for learned diagnostic state."""

from datetime import date, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from .config import FeatureScalar


def encode_scalar(value: FeatureScalar | None) -> dict[str, object]:
    if value is None:
        return {"type": "none"}
    if isinstance(value, datetime):
        return {
            "type": "datetime",
            "value": value.isoformat(),
            "timezone": getattr(value.tzinfo, "key", None),
            "fold": value.fold,
        }
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    return {"type": "str", "value": value}


def mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return cast(dict[str, Any], value)


def decode_scalar(payload: object) -> FeatureScalar | None:
    data = mapping(payload)
    kind = data["type"]
    value = data.get("value")
    if kind == "none":
        return None
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "int" and type(value) is int:
        return value
    if kind == "str" and isinstance(value, str):
        return value
    if isinstance(value, str):
        if kind == "float":
            return float.fromhex(value)
        if kind == "date":
            return date.fromisoformat(value)
        if kind == "datetime":
            result = datetime.fromisoformat(value)
            zone = data.get("timezone")
            if zone is not None:
                if not isinstance(zone, str) or result.tzinfo is None:
                    raise ValueError("Invalid datetime timezone")
                result = result.astimezone(ZoneInfo(zone))
            fold = data.get("fold", 0)
            if type(fold) is not int or fold not in (0, 1):
                raise ValueError("Invalid datetime fold")
            return result.replace(fold=fold)
    raise ValueError(f"Invalid typed scalar: {kind!r}")
