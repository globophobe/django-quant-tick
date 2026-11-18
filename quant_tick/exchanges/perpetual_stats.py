import logging
from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd
from django.db import transaction
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
from .hyperliquid.api import normalize_coin
from .phoenix.constants import MARKET_HISTORY_INTERVAL as PHOENIX_INTERVAL
from .phoenix.history import PhoenixPositionHistory
from .phoenix.market_history import phoenix_market_history
from .quicknode.api import available as quicknode_available
from .quicknode.market_history import quicknode_market_history
from .quicknode.solana import available as solana_available

logger = logging.getLogger(__name__)

PERPETUAL_STATS_FETCH_WINDOW = timedelta(days=90)
PERPETUAL_STATS_MIN_PERSISTENCE_INTERVAL_MINUTES = 60
PERPETUAL_STATS_INTERVALS = {
    Exchange.BINANCE_FUTURES: BINANCE_INTERVAL,
    Exchange.BYBIT_LINEAR: BYBIT_INTERVAL,
    Exchange.BYBIT_INVERSE: BYBIT_INTERVAL,
    Exchange.PHOENIX: PHOENIX_INTERVAL,
    Exchange.HYPERLIQUID: "1h",
}
PERPETUAL_STATS_SUPPORTED_EXCHANGES = frozenset(PERPETUAL_STATS_INTERVALS)
PERPETUAL_STATS_BOUNDARY_SNAPSHOT_EXCHANGES = frozenset(
    {
        Exchange.BYBIT_LINEAR,
        Exchange.BYBIT_INVERSE,
    }
)


def enabled_perpetual_stats_exchanges():
    if quicknode_available():
        return PERPETUAL_STATS_SUPPORTED_EXCHANGES
    return PERPETUAL_STATS_SUPPORTED_EXCHANGES - {Exchange.HYPERLIQUID}


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
    *,
    position_history=None,
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
    if symbol.exchange == Exchange.PHOENIX:
        return phoenix_market_history(
            symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            position_history=position_history,
        )
    if symbol.exchange == Exchange.HYPERLIQUID:
        return quicknode_market_history(
            normalize_coin(symbol.api_symbol), timestamp_from, timestamp_to
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


def perpetual_stats_persistence_end(
    symbol: Symbol,
    timestamp_to: datetime,
    *,
    frequency: int,
    persistence_frequency: int,
) -> datetime:
    """Return the exclusive end covering observations available by timestamp_to."""
    if symbol.exchange in PERPETUAL_STATS_BOUNDARY_SNAPSHOT_EXCHANGES:
        return pd.Timestamp(timestamp_to).ceil(f"{frequency}min").to_pydatetime()
    return get_complete_interval_end(timestamp_to, persistence_frequency)


def perpetual_stats(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
    *,
    assert_lease_owned: Callable[[], None] | None = None,
) -> None:
    """Fetch and persist native-cadence perpetual stats observations.

    Ordinary backfills stop on an empty window only when its chunk and all
    earlier history have no stored observations, including partial rows. Retries
    scan the requested range unless the adapter explicitly reports exhaustion.
    """
    if symbol.exchange == Exchange.HYPERLIQUID and not quicknode_available():
        return
    timestamp_range = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if timestamp_range is None:
        return
    timestamp_from, timestamp_to = timestamp_range
    requested_timestamp_to = timestamp_to
    frequency = perpetual_stats_frequency(symbol)
    persistence_frequency = max(
        frequency,
        PERPETUAL_STATS_MIN_PERSISTENCE_INTERVAL_MINUTES,
    )
    if persistence_frequency % frequency:
        raise ValueError(
            "Perpetual stats native frequency must divide its persistence interval."
        )
    timestamp_from = (
        pd.Timestamp(timestamp_from).ceil(f"{frequency}min").to_pydatetime()
    )
    timestamp_to = perpetual_stats_persistence_end(
        symbol,
        timestamp_to,
        frequency=frequency,
        persistence_frequency=persistence_frequency,
    )
    if timestamp_to <= timestamp_from:
        return

    fetch_window = PERPETUAL_STATS_FETCH_WINDOW
    position_history = None
    # Persist bounded paid windows before advancing, so a later provider failure
    # does not discard an entire archive chunk's successful requests.
    if symbol.exchange == Exchange.PHOENIX and solana_available():
        fetch_window = timedelta(hours=1)
        position_history = PhoenixPositionHistory(
            symbol, assert_lease_owned=assert_lease_owned
        )
    elif symbol.exchange == Exchange.HYPERLIQUID:
        fetch_window = timedelta(hours=6)
    for chunk_from, chunk_to in iter_chunks(
        timestamp_from,
        timestamp_to,
        value=fetch_window,
        reverse=True,
    ):
        windows = [(chunk_from, chunk_to)]
        if not retry:
            queryset = PerpetualStatsData.objects.in_range(
                symbol,
                frequency,
                chunk_from,
                chunk_to,
            ).complete_for_exchange(symbol.exchange)
            if position_history is not None:
                queryset = queryset.filter(
                    json_data__cohort_replay_validated=True,
                    json_data__trade_position_flow_validated=True,
                )
            existing = list(queryset.values_list("timestamp", flat=True))
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
            options = (
                {"position_history": position_history}
                if position_history is not None
                else {}
            )
            request_to = min(fetch_to, requested_timestamp_to)
            logger.info(
                "%s: perpetual stats %s to %s",
                symbol,
                fetch_from,
                request_to,
            )
            df = perpetual_stats_api(
                symbol,
                fetch_from,
                request_to,
                **options,
            )
            history_exhausted = df.attrs.get(HISTORY_EXHAUSTED_ATTR)
            df = validate_perpetual_stats_frame(symbol, df)
            with transaction.atomic():
                PerpetualStatsData.write(
                    symbol,
                    frequency,
                    fetch_from,
                    fetch_to,
                    df,
                    assert_lease_owned=assert_lease_owned,
                )
                if position_history is not None:
                    position_history.commit()
            if history_exhausted is True:
                return
            if (
                not retry
                and history_exhausted is not False
                and df.empty
                and not PerpetualStatsData.objects.filter(
                    symbol=symbol,
                    frequency=frequency,
                    timestamp__lt=chunk_to,
                ).exists()
            ):
                return
