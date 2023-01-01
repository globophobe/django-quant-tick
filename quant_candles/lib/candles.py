from datetime import datetime
from decimal import Decimal
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from .aggregate import aggregate_sum
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
    # Maybe no candles
    if len(data_frame) > 0:
        # Assert timestamp_from <= data_frame.timestamp < timestamp_to
        is_less_than_timestamp_from = (
            len(data_frame[data_frame.timestamp < timestamp_from]) > 0
        )
        is_greater_than_timestamp_to = (
            len(data_frame[data_frame.timestamp >= timestamp_to]) > 0
        )
        try:
            assert not is_less_than_timestamp_from and not is_greater_than_timestamp_to
        except AssertionError as e:
            if len(data_frame) == 1:
                # Assert data_frame.timestamp == timestamp_from == timestamp_to
                assert len(data_frame[data_frame.timestamp == timestamp_from]) == 1
                assert len(data_frame[data_frame.timestamp == timestamp_to]) == 1
            else:
                raise e
        finally:
            df = data_frame.set_index("timestamp")
            # REST API, data is reverse order
            if reverse:
                df = df.iloc[::-1]
            return df
    return data_frame


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
        if len(data_frame):
            capitalized_key = key.capitalize()
            total_key = f"total{capitalized_key}"
            # If there was a significant trade filter, total_key
            attrs = total_key if total_key in data_frame.columns else key
            df = aggregate_sum(data_frame, attrs=attrs, window="1t")
            for row in df.itertuples():
                k = key.title()
                timestamp = row.Index
                try:
                    candle = candles.loc[timestamp]
                # Candle may be missing from API result
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
    # Candle and trade API data availability may differ
    else:
        validated = {
            timestamp: None
            for timestamp in get_range(timestamp_from, timestamp_to)
            if timestamp >= timestamp_from and timestamp < timestamp_to
        }
    return validated
