from datetime import timezone
from decimal import Decimal

import numpy as np


def utc_timestamp(data_frame):
    # Because pyarrow.lib.ArrowInvalid: Casting from timestamp[ns]
    # to timestamp[us, tz=UTC] would lose data.
    data_frame.timestamp = data_frame.apply(
        lambda x: x.timestamp.tz_localize(timezone.utc), axis=1
    )
    return data_frame


def calculate_notional(data_frame):
    data_frame["notional"] = data_frame.apply(lambda x: x.volume / x.price, axis=1)
    return data_frame


def calculate_tick_rule(data_frame):
    data_frame["tickRule"] = data_frame.apply(
        lambda x: (1 if x.tickDirection in ("PlusTick", "ZeroPlusTick") else -1),
        axis=1,
    )
    return data_frame


def set_dtypes(data_frame):
    df = data_frame.astype({"index": "int64"})
    for column in ("price", "volume"):
        df = set_type_decimal(df, column)
    return df


def set_type_decimal(data_frame, column):
    data_frame[column] = data_frame[column].apply(Decimal)
    return data_frame


def strip_nanoseconds(data_frame):
    # Bitmex data is accurate to the nanosecond.
    # However, data is typically only provided to the microsecond.
    data_frame["nanoseconds"] = data_frame.apply(
        lambda x: x.timestamp.nanosecond, axis=1
    )
    data_frame.timestamp = data_frame.apply(
        lambda x: x.timestamp.replace(nanosecond=0)
        if x.nanoseconds > 0
        else x.timestamp,
        axis=1,
    )
    return data_frame


def assert_type_decimal(data_frame, columns):
    for column in columns:
        data_frame[column].apply(lambda x: assert_decimal(x))


def assert_decimal(x):
    assert isinstance(x, Decimal)


def is_decimal_close(d1: Decimal, d2: Decimal) -> None:
    """Is decimal one close to decimal two?"""
    return np.isclose(float(d1), float(d2))
