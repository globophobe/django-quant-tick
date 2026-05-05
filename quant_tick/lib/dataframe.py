from collections.abc import Iterable
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame


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


def is_decimal_close(d1: Decimal, d2: Decimal) -> bool:
    """Is decimal one close to decimal two?"""
    return np.isclose(float(d1), float(d2))
