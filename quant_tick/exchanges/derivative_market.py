from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import Exchange, SymbolType
from quant_tick.lib import (
    get_complete_interval_end,
    iter_chunks,
    iter_missing,
    parse_fixed_resolution_minutes,
)
from quant_tick.models import DerivativeMarketData, Symbol
from quant_tick.models.derivative_market import derivative_market_required_fields

from .binance_futures.constants import MARKET_HISTORY_INTERVAL as BINANCE_INTERVAL
from .binance_futures.market_history import binance_market_history
from .bybit.candles import get_bybit_category
from .bybit.constants import MARKET_HISTORY_INTERVAL as BYBIT_INTERVAL
from .bybit.funding import bybit_market_history

DERIVATIVE_MARKET_FETCH_WINDOW = timedelta(days=90)
DERIVATIVE_MARKET_MIN_PERSISTENCE_INTERVAL_MINUTES = 60
DERIVATIVE_MARKET_INTERVALS = {
    Exchange.BINANCE_FUTURES: BINANCE_INTERVAL,
    Exchange.BYBIT_LINEAR: BYBIT_INTERVAL,
    Exchange.BYBIT_INVERSE: BYBIT_INTERVAL,
}
DERIVATIVE_MARKET_SUPPORTED_EXCHANGES = frozenset(DERIVATIVE_MARKET_INTERVALS)


def derivative_market_frequency(symbol: Symbol) -> int:
    """Return the native stored interval in minutes for a supported symbol."""
    if symbol.symbol_type != SymbolType.PERPETUAL:
        raise ValueError("Derivative market data is only available for perpetuals.")
    try:
        interval = DERIVATIVE_MARKET_INTERVALS[symbol.exchange]
    except KeyError as exc:
        raise NotImplementedError(
            f"Derivative market data is not implemented for {symbol.exchange}."
        ) from exc
    return parse_fixed_resolution_minutes(interval)


def derivative_market_data_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch native-cadence derivative market observations."""
    if symbol.symbol_type != SymbolType.PERPETUAL:
        raise ValueError("Derivative market data is only available for perpetuals.")
    if symbol.exchange == Exchange.BINANCE_FUTURES:
        return binance_market_history(
            symbol.api_symbol,
            timestamp_from,
            timestamp_to,
        )
    if symbol.exchange in {Exchange.BYBIT_LINEAR, Exchange.BYBIT_INVERSE}:
        return bybit_market_history(
            symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            category=get_bybit_category(symbol.exchange),
        )
    raise NotImplementedError(
        f"Derivative market data is not implemented for {symbol.exchange}."
    )


def complete_derivative_market_frame(symbol: Symbol, df: DataFrame) -> DataFrame:
    """Keep only observations populated by every required venue endpoint."""
    if df.empty:
        return df
    required = derivative_market_required_fields(symbol.exchange)
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(
            f"Derivative market response for {symbol.exchange} omitted required "
            f"columns: {missing}"
        )
    return df.dropna(subset=list(required))


def derivative_market_data(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
    *,
    assert_lease_owned: Callable[[], None] | None = None,
) -> None:
    """Fetch and persist native-cadence derivative market observations."""
    timestamp_range = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if timestamp_range is None:
        return
    timestamp_from, timestamp_to = timestamp_range
    frequency = derivative_market_frequency(symbol)
    persistence_frequency = max(
        frequency,
        DERIVATIVE_MARKET_MIN_PERSISTENCE_INTERVAL_MINUTES,
    )
    if persistence_frequency % frequency:
        raise ValueError(
            "Derivative market native frequency must divide its persistence interval."
        )
    timestamp_from = pd.Timestamp(timestamp_from).ceil(
        f"{frequency}min"
    ).to_pydatetime()
    timestamp_to = get_complete_interval_end(timestamp_to, persistence_frequency)
    if timestamp_to <= timestamp_from:
        return

    for chunk_from, chunk_to in iter_chunks(
        timestamp_from,
        timestamp_to,
        value=DERIVATIVE_MARKET_FETCH_WINDOW,
        reverse=True,
    ):
        windows = [(chunk_from, chunk_to)]
        if not retry:
            existing = list(
                DerivativeMarketData.objects.in_range(
                    symbol,
                    frequency,
                    chunk_from,
                    chunk_to,
                )
                .complete_for_exchange(symbol.exchange)
                .values_list("timestamp", flat=True)
            )
            windows = list(
                iter_missing(
                    chunk_from,
                    chunk_to,
                    existing,
                    reverse=True,
                    value=f"{frequency}min",
                )
            )

        for fetch_from, fetch_to in windows:
            df = derivative_market_data_api(symbol, fetch_from, fetch_to)
            df = complete_derivative_market_frame(symbol, df)
            DerivativeMarketData.write(
                symbol,
                frequency,
                fetch_from,
                fetch_to,
                df,
                assert_lease_owned=assert_lease_owned,
            )
