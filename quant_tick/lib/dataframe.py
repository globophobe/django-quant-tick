from collections.abc import Iterable
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.constants import ZERO

DECIMAL_CLOSE_EPSILON = Decimal("1e-8")


def calculate_notional(data_frame: DataFrame) -> DataFrame:
    """Calculate notional."""
    data_frame["notional"] = data_frame["volume"] / data_frame["price"]
    return data_frame


def calculate_tick_rule(data_frame: DataFrame) -> DataFrame:
    """Calculate tick rule."""
    data_frame["tickRule"] = np.where(
        data_frame["tickDirection"].isin(("PlusTick", "ZeroPlusTick")), 1, -1
    )
    return data_frame


def set_dtypes(data_frame: DataFrame) -> DataFrame:
    """Set dtypes."""
    for column in ("price", "volume"):
        data_frame = set_type_decimal(data_frame, column)
    return data_frame


def set_type_decimal(data_frame: DataFrame, column: str) -> DataFrame:
    """Set type decimal."""
    data_frame[column] = data_frame[column].map(Decimal)
    return data_frame


def to_decimal_or_none(value: object) -> Decimal | None:
    """Normalize nullable numeric DataFrame values to Decimal."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def normalize_timestamp_data_frame(data_frame: DataFrame) -> DataFrame:
    """Return a DataFrame with timestamp as a regular column."""
    frame = data_frame.reset_index()
    if "timestamp" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    return frame


def assert_type_decimal(data_frame: DataFrame, columns: Iterable[str]) -> None:
    """Assert type decimal."""
    for column in columns:
        assert all(isinstance(value, Decimal) for value in data_frame[column])


def is_decimal_close(
    d1: object,
    d2: object,
    *,
    epsilon: Decimal = DECIMAL_CLOSE_EPSILON,
) -> bool:
    """Return whether two decimals differ by no more than an absolute epsilon."""
    left = to_decimal_or_none(d1)
    right = to_decimal_or_none(d2)
    if left is None or right is None:
        return left == right
    return abs(left - right) <= epsilon


def has_column_group(data_frame: DataFrame, columns: Iterable[str]) -> bool:
    """Return whether a related column group is present, rejecting partial groups."""
    required = set(columns)
    available = set(data_frame.columns)
    if not required & available:
        return False

    missing = required - available
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {names}.")
    return True


def get_frame_totals(data_frame: DataFrame | None) -> tuple[Decimal, Decimal]:
    """Return volume and notional totals for raw or totalized trade frames."""
    if data_frame is None or data_frame.empty:
        return ZERO, ZERO
    if has_column_group(data_frame, ("totalVolume", "totalNotional")):
        return (
            data_frame.totalVolume.sum() or ZERO,
            data_frame.totalNotional.sum() or ZERO,
        )
    return data_frame.volume.sum() or ZERO, data_frame.notional.sum() or ZERO


def validate_totals(**frames: DataFrame | None) -> None:
    """Validate that non-null frames carry equal volume and notional totals."""
    totals = [
        (name, get_frame_totals(data_frame))
        for name, data_frame in frames.items()
        if data_frame is not None
    ]
    if len(totals) < 2:
        return

    base_name, (base_volume, base_notional) = totals[0]
    for name, (volume, notional) in totals[1:]:
        if not is_decimal_close(volume, base_volume):
            raise ValueError(f"{name} volume does not match {base_name}.")
        if not is_decimal_close(notional, base_notional):
            raise ValueError(f"{name} notional does not match {base_name}.")
