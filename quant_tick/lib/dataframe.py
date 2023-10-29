from decimal import Decimal
from typing import Any, Iterable

import numpy as np
from pandas import DataFrame


def calculate_notional(data_frame: DataFrame) -> DataFrame:
    """Calculate notional."""
    data_frame["notional"] = data_frame.apply(lambda x: x.volume / x.price, axis=1)
    return data_frame


def calculate_tick_rule(data_frame: DataFrame) -> DataFrame:
    """Calculate tick rule."""
    data_frame["tickRule"] = data_frame.apply(
        lambda x: (1 if x.tickDirection in ("PlusTick", "ZeroPlusTick") else -1),
        axis=1,
    )
    return data_frame


def set_dtypes(data_frame: DataFrame) -> DataFrame:
    """Set dtypes."""
    for column in ("price", "volume"):
        data_frame = set_type_decimal(data_frame, column)
    return data_frame


def set_type_decimal(data_frame: DataFrame, column: str) -> DataFrame:
    """Set type decimal."""
    data_frame[column] = data_frame[column].apply(Decimal)
    return data_frame


def assert_type_decimal(data_frame: DataFrame, columns: Iterable[str]) -> None:
    """Assert type decimal."""
    for column in columns:
        data_frame[column].apply(lambda x: assert_decimal(x))


def assert_decimal(x: Any) -> None:
    """Assert decimal."""
    assert isinstance(x, Decimal)


def is_decimal_close(d1: Decimal, d2: Decimal) -> bool:
    """Is decimal one close to decimal two?"""
    return np.isclose(float(d1), float(d2))
