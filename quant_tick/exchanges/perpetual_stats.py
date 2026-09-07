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
from quant_tick.models import PerpetualStatsData, Symbol
from quant_tick.models.perpetual_stats import perpetual_stats_required_fields

from .binance_futures.constants import MARKET_HISTORY_INTERVAL as BINANCE_INTERVAL
from .binance_futures.market_history import (
    HISTORY_EXHAUSTED_ATTR,
    binance_market_history,
)
from .bybit.candles import get_bybit_category
from .bybit.constants import MARKET_HISTORY_INTERVAL as BYBIT_INTERVAL
from .bybit.funding import bybit_market_history

PERPETUAL_STATS_FETCH_WINDOW = timedelta(days=90)
PERPETUAL_STATS_MIN_PERSISTENCE_INTERVAL_MINUTES = 60
PERPETUAL_STATS_INTERVALS = {
    Exchange.BINANCE_FUTURES: BINANCE_INTERVAL,
    Exchange.BYBIT_LINEAR: BYBIT_INTERVAL,
    Exchange.BYBIT_INVERSE: BYBIT_INTERVAL,
}
PERPETUAL_STATS_SUPPORTED_EXCHANGES = frozenset(PERPETUAL_STATS_INTERVALS)


def perpetual_stats_frequency(symbol: Symbol) -> int:
    """Return the native stored interval in minutes for a supported symbol."""
    if symbol.symbol_type != SymbolType.PERPETUAL:
        raise ValueError("Perpetual stats data is only available for perpetuals.")
    try:
        interval = PERPETUAL_STATS_INTERVALS[symbol.exchange]
    except KeyError as exc:
        raise NotImplementedError(
            f"Perpetual stats data is not implemented for {symbol.exchange}."
        ) from exc
    return parse_fixed_resolution_minutes(interval)


def perpetual_stats_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch native-cadence perpetual stats observations."""
    if symbol.symbol_type != SymbolType.PERPETUAL:
        raise ValueError("Perpetual stats data is only available for perpetuals.")
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
        f"Perpetual stats data is not implemented for {symbol.exchange}."
    )


def complete_perpetual_stats_frame(symbol: Symbol, df: DataFrame) -> DataFrame:
    """Keep only observations populated by every required venue endpoint."""
    df = validate_perpetual_stats_frame(symbol, df)
    if df.empty:
        return df
    required = perpetual_stats_required_fields(symbol.exchange)
    return df.dropna(subset=list(required))


def validate_perpetual_stats_frame(symbol: Symbol, df: DataFrame) -> DataFrame:
    """Require the adapter schema while retaining partial observations."""
    if df.empty:
        return df
    required = perpetual_stats_required_fields(symbol.exchange)
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(
            f"Perpetual stats response for {symbol.exchange} omitted required "
            f"columns: {missing}"
        )
    return df


def perpetual_stats(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
    *,
    assert_lease_owned: Callable[[], None] | None = None,
) -> None:
    """Fetch and persist native-cadence perpetual stats observations."""
    timestamp_range = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if timestamp_range is None:
        return
    timestamp_from, timestamp_to = timestamp_range
    frequency = perpetual_stats_frequency(symbol)
    persistence_frequency = max(
        frequency,
        PERPETUAL_STATS_MIN_PERSISTENCE_INTERVAL_MINUTES,
    )
    if persistence_frequency % frequency:
        raise ValueError(
            "Perpetual stats native frequency must divide its persistence interval."
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
        value=PERPETUAL_STATS_FETCH_WINDOW,
        reverse=True,
    ):
        windows = [(chunk_from, chunk_to)]
        if not retry:
            existing = list(
                PerpetualStatsData.objects.in_range(
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
            df = perpetual_stats_api(symbol, fetch_from, fetch_to)
            history_exhausted = df.attrs.get(HISTORY_EXHAUSTED_ATTR) is True
            df = validate_perpetual_stats_frame(symbol, df)
            PerpetualStatsData.write(
                symbol,
                frequency,
                fetch_from,
                fetch_to,
                df,
                assert_lease_owned=assert_lease_owned,
            )
            if history_exhausted:
                return
