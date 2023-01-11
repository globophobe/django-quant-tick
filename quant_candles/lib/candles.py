from datetime import datetime
from decimal import Decimal
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from .aggregate import aggregate_sum, filter_by_timestamp
from .calendar import get_range
from .dataframe import is_decimal_close


def candles_to_data_frame(
    timestamp_from: datetime,
    timestamp_to: datetime,
    candles: List[dict],
    reverse: bool = True,
) -> DataFrame:
    """Get candle data_frame."""
    data_frame = pd.DataFrame(candles)
    df = filter_by_timestamp(
        data_frame,
        timestamp_from,
        timestamp_to,
        inclusive=timestamp_from == timestamp_to,
    )
    if len(df):
        df.set_index("timestamp", inplace=True)
    # REST API, data is reverse order.
    return df.iloc[::-1] if reverse else df


def validate_data_frame(
    timestamp_from: datetime,
    timestamp_to: datetime,
    data_frame: DataFrame,
    candles: DataFrame,
    should_aggregate_trades: bool,
) -> Optional[dict]:
    """Validate data_frame with candles from Exchange API."""
    if len(candles):
        if "notional" in candles.columns:
            key = "notional"
        elif "volume" in candles.columns:
            key = "volume"
        else:
            raise NotImplementedError
        validated = {
            candle.Index: True
            if (value := getattr(candle, key)) == Decimal("0")
            else value
            for candle in candles.itertuples()
        }
        k = key.title()
        capitalized_key = key.capitalize()
        total_key = f"total{capitalized_key}"
        if len(data_frame):
            # If there was a significant trade filter, total_key
            attrs = total_key if total_key in data_frame.columns else key
            df = aggregate_sum(data_frame, attrs=attrs, window="1t")
            for row in df.itertuples():
                timestamp = row.Index
                try:
                    candle = candles.loc[timestamp]
                # Candle may be missing from API result.
                except KeyError:
                    validated[timestamp] = None
                else:
                    values = row[1], candle[key]
                    is_close = is_decimal_close(*values)
                    if is_close:
                        validated[timestamp] = True
                    else:
                        # Maybe int64
                        if isinstance(candle[key], np.int64):
                            v = int(candle[key])
                        else:
                            v = candle[key]
                        validated[timestamp] = {key: row[1], f"exchange{k}": v}
        # Candle and trade API data availability may differ.
        for timestamp, v in validated.items():
            if isinstance(v, Decimal):
                validated[timestamp] = {key: Decimal("0"), f"exchange{k}": v}
    else:
        validated = {
            timestamp: None
            for timestamp in get_range(timestamp_from, timestamp_to)
            if timestamp >= timestamp_from and timestamp < timestamp_to
        }
    return validated


def sum_validation(data: List[dict]) -> dict:
    """Sum validation."""
    validation = {}
    for d in data:
        for key, value in d.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    validation.setdefault(k, v)
                    validation[k] += v
            elif value is None:
                validation.setdefault(None, 0)
                validation[None] += 1
    return validation
