from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from .aggregate import filter_by_timestamp
from .calendar import iter_window
from .dataframe import is_decimal_close
from .experimental import calc_notional_exponent, calc_volume_exponent

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
    min_volume_exponent: int = 2,
    min_notional_exponent: int = 1,
) -> dict:
    """Aggregate candle."""
    ts = timestamp if timestamp else data_frame.iloc[0].timestamp
    data = {"timestamp": ts}
    data.update(_aggregate_ohlc(data_frame))
    data.update(
        _aggregate_totals(data_frame, min_volume_exponent, min_notional_exponent)
    )
    data.update(_aggregate_realized_variance(data_frame))
    return data


def _aggregate_ohlc(df: DataFrame) -> dict:
    """Aggregate OHLC from trade data."""
    return {
        "open": df.iloc[0].price,
        "high": df.price.max(),
        "low": df.price.min(),
        "close": df.iloc[-1].price,
    }


def _aggregate_totals(
    df: DataFrame,
    min_volume_exponent: int = 2,
    min_notional_exponent: int = 1,
) -> dict:
    """Aggregate volume, notional, ticks, and round fields."""
    data = {}
    has_totals = "totalVolume" in df.columns
    is_buy = df.tickRule == 1

    if has_totals:
        data["volume"] = df.totalVolume.sum()
        data["buyVolume"] = df.totalBuyVolume.sum()
        data["notional"] = df.totalNotional.sum()
        data["buyNotional"] = df.totalBuyNotional.sum()
        data["ticks"] = int(df.totalTicks.sum())
        data["buyTicks"] = int(df.totalBuyTicks.sum())
    else:
        data["volume"] = df.volume.sum()
        data["buyVolume"] = df.loc[is_buy, "volume"].sum()
        data["notional"] = df.notional.sum()
        data["buyNotional"] = df.loc[is_buy, "notional"].sum()
        if "ticks" in df.columns:
            data["ticks"] = int(df.ticks.sum())
            data["buyTicks"] = int(df.loc[is_buy, "ticks"].sum())
        else:
            data["ticks"] = len(df)
            data["buyTicks"] = int(is_buy.sum())
    # Round volume
    volume_exps = df.volume.apply(calc_volume_exponent)
    is_round_volume = volume_exps >= min_volume_exponent
    round_vol = df.loc[is_round_volume]
    round_buy_vol = df.loc[is_round_volume & is_buy]
    data["roundVolume"] = round_vol.volume.sum() if len(round_vol) else ZERO
    data["roundBuyVolume"] = round_buy_vol.volume.sum() if len(round_buy_vol) else ZERO
    # Round notional
    notional_exps = df.notional.apply(calc_notional_exponent)
    is_round_notional = notional_exps >= min_notional_exponent
    round_not = df.loc[is_round_notional]
    round_buy_not = df.loc[is_round_notional & is_buy]
    data["roundNotional"] = round_not.notional.sum() if len(round_not) else ZERO
    data["roundBuyNotional"] = round_buy_not.notional.sum() if len(round_buy_not) else ZERO
    return data


def _aggregate_realized_variance(df: DataFrame) -> dict:
    """Compute realized variance."""
    if "price" not in df.columns or len(df) == 0:
        return {"realizedVariance": ZERO}

    if len(df) > 1:
        prices = df.price.astype(float)
        log_prices = np.log(prices)
        log_returns = log_prices.diff().dropna()
        return {"realizedVariance": Decimal(str((log_returns**2).sum()))}

    return {"realizedVariance": ZERO}


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
