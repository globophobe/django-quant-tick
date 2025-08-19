from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from .aggregate import filter_by_timestamp
from .calendar import iter_window
from .dataframe import is_decimal_close

ZERO = Decimal("0")


def candles_to_data_frame(
    timestamp_from: datetime,
    timestamp_to: datetime,
    candles: list[dict],
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


def aggregate_candles(
    data_frame: DataFrame,
    timestamp_from: datetime,
    timestamp_to: datetime,
    window: str = "1min",
    as_data_frame: bool = True,
) -> list[dict]:
    """Aggregate candles"""
    data = []
    for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, window):
        df = filter_by_timestamp(data_frame, ts_from, ts_to)
        if len(df):
            candle = aggregate_candle(df, timestamp=ts_from)
            data.append(candle)
        else:
            d = {"timestamp": ts_from}
            for k in "open", "high", "low", "close":
                d[k] = None
            for k in (
                "volume",
                "buyVolume",
                "notional",
                "buyNotional",
                "ticks",
                "buyTicks",
            ):
                d[k] = ZERO
    if as_data_frame:
        df = DataFrame(data)
        if len(df):
            df.set_index("timestamp", inplace=True)
        return df
    else:
        return data


def aggregate_candle(
    data_frame: DataFrame,
    timestamp: datetime | None = None,
    quantiles: list[int] | None = None,
) -> dict:
    """Aggregate candle"""
    quantiles = quantiles or []

    first_row = data_frame.iloc[0]
    last_row = data_frame.iloc[-1]
    if "open" in data_frame.columns:
        open_price = first_row.open
    else:
        open_price = first_row.price
    if "close" in data_frame.columns:
        close_price = last_row.close
    else:
        close_price = last_row.price
    if "high" in data_frame.columns:
        high = data_frame.high.max()
    else:
        high = data_frame.price.max()
    if "low" in data_frame.columns:
        low = data_frame.low.min()
    else:
        low = data_frame.price.min()
    data = {
        "timestamp": timestamp if timestamp else first_row.timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close_price,
    }
    buy_data_frame = data_frame[data_frame.tickRule == 1]
    data["volume"] = data_frame.volume.sum()
    data["buyVolume"] = buy_data_frame.volume.sum()
    for column in ("totalVolume", "totalBuyVolume"):
        if column in data_frame.columns:
            data[column] = data_frame[column].sum()
    data["notional"] = data_frame.notional.sum()
    data["buyNotional"] = buy_data_frame.notional.sum()
    for column in ("totalNotional", "totalBuyNotional"):
        if column in data_frame.columns:
            data[column] = data_frame[column].sum()
    data["ticks"] = int(
        data_frame.ticks.sum() if "ticks" in data_frame.columns else len(data_frame)
    )
    data["buyTicks"] = int(
        buy_data_frame.ticks.sum()
        if "ticks" in buy_data_frame.columns
        else len(buy_data_frame)
    )
    for column in ("totalTicks", "totalBuyTicks"):
        if column in data_frame.columns:
            data[column] = int(data_frame[column].sum())
    if quantiles:
        df = data_frame[data_frame.notional.notna()]
        if len(df):
            quants = [q / 100.0 for q in sorted(quantiles, reverse=True)]
            numeric_notional = pd.to_numeric(df.notional)
            thresholds = [numeric_notional.quantile(q) for q in quants]
            for index, q in enumerate(quants):
                if index == 0:
                    q_df = df[numeric_notional >= thresholds[index]]
                else:
                    q_df = df[
                        (numeric_notional >= thresholds[index])
                        & (numeric_notional < thresholds[index - 1])
                    ]
                if len(q_df):
                    q_buy_df = q_df[q_df.tickRule == 1]
                    key = int(q * 100)
                    data[f"q{key}Volume"] = q_df.volume.sum()
                    data[f"q{key}BuyVolume"] = q_buy_df.volume.sum()
                    data[f"q{key}Notional"] = q_df.notional.sum()
                    data[f"q{key}BuyNotional"] = q_buy_df.notional.sum()
                    data[f"q{key}Ticks"] = int(q_df.ticks.sum())
                    data[f"q{key}BuyTicks"] = int(q_buy_df.ticks.sum())
    return data


def validate_aggregated_candles(
    aggregated_candles: DataFrame, exchange_candles: DataFrame
) -> tuple[DataFrame, bool | None]:
    """Validate data_frame with candles from Exchange API."""
    ok = None
    aggregated_candles["validated"] = pd.Series()
    if len(exchange_candles):
        if "notional" in exchange_candles.columns:
            key = "notional"
        elif "volume" in exchange_candles.columns:
            key = "volume"
        else:
            raise NotImplementedError
        k = key.title()
        exchange_key = f"exchange{k}"
        if len(aggregated_candles):
            aggregated_candles.insert(
                aggregated_candles.columns.get_loc(key), exchange_key, pd.Series()
            )
            for row in aggregated_candles.itertuples():
                timestamp = row.Index
                try:
                    candle = exchange_candles.loc[timestamp]
                # Candle may be missing from API result.
                except KeyError:
                    pass
                else:
                    value = getattr(candle, key)
                    if isinstance(value, int):
                        value = Decimal(value)
                    aggregated_candles.at[row.Index, exchange_key] = value
                    aggregated_candles.at[row.Index, "validated"] = is_decimal_close(
                        getattr(row, key), value
                    )
            all_true = all(aggregated_candles.validated.eq(True))
            some_false = any(aggregated_candles.validated.eq(False))
            some_none = any(aggregated_candles.validated.isna())
            if all_true:
                ok = True
            elif some_false or some_none:
                if some_false:
                    ok = False
                else:
                    ok = None
            else:
                raise NotImplementedError

        elif exchange_candles[key].sum() == 0:
            ok = True

        if ok in (True, None):
            # Maybe candle with no volume or notional.
            missing = exchange_candles.index.difference(aggregated_candles.index)
            if len(missing):
                # If missing candles have volume or notional, then ok should be False.
                # Maybe validates on retry.
                if (
                    exchange_candles[exchange_candles.index.isin(missing)][key].sum()
                    != 0
                ):
                    ok = False

    return aggregated_candles, ok
