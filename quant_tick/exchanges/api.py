from collections.abc import Callable
from datetime import datetime, timedelta

from pandas import DataFrame

from quant_tick.constants import Exchange, SymbolType
from quant_tick.lib import iter_chunks, parse_fixed_resolution_minutes
from quant_tick.models import ExchangeCandleData, FundingData, Symbol, TradeData

from .binance import binance_candles, binance_funding, binance_trades
from .bitfinex import bitfinex_candles, bitfinex_trades
from .bitmex import bitmex_candles, bitmex_funding, bitmex_trades
from .coinbase import coinbase_candles, coinbase_trades
from .coinbase_advanced import (
    coinbase_advanced_candles,
    coinbase_advanced_funding,
    coinbase_advanced_trades,
)
from .hyperliquid import hyperliquid_candles, hyperliquid_funding

FUNDING_FETCH_WINDOW = timedelta(days=90)
FUNDING_CHUNKED_EXCHANGES = {
    Exchange.BINANCE_FUTURES,
    Exchange.BITMEX,
    Exchange.HYPERLIQUID,
}


def get_binance_symbol_type(symbol: Symbol) -> str:
    expected = (
        SymbolType.PERPETUAL
        if symbol.exchange == Exchange.BINANCE_FUTURES
        else SymbolType.SPOT
    )
    if symbol.symbol_type == expected:
        return expected
    if symbol.exchange == Exchange.BINANCE_FUTURES:
        raise ValueError("binance-futures must be perpetuals.")
    raise ValueError("binance symbols must be spot.")


def api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
    verbose: bool = False,
) -> None:
    """Fetch exchange trades and persist TradeData rows."""

    def on_data_frame(
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        trades: DataFrame,
        candles: DataFrame,
    ) -> DataFrame:
        """Write one fetched trade slice."""
        TradeData.write(symbol, timestamp_from, timestamp_to, trades, candles)

    trades_api(
        symbol=symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    )


def trades_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
) -> None:
    """Dispatch trade fetching to the exchange-specific adapter."""
    timestamp_range = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if timestamp_range is None:
        return
    timestamp_from, timestamp_to = timestamp_range
    exchange = symbol.exchange
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": on_data_frame,
        "retry": retry,
        "verbose": verbose,
    }
    if exchange in (Exchange.BINANCE, Exchange.BINANCE_FUTURES):
        get_binance_symbol_type(symbol)
        binance_trades(symbol, **kwargs)
    elif exchange == Exchange.BITFINEX:
        bitfinex_trades(symbol, **kwargs)
    elif exchange == Exchange.BITMEX:
        bitmex_trades(symbol, **kwargs)
    elif exchange == Exchange.COINBASE:
        coinbase_trades(symbol, **kwargs)
    elif exchange == Exchange.COINBASE_ADVANCED:
        coinbase_advanced_trades(symbol, **kwargs)


def candles_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = None,
) -> DataFrame:
    """Dispatch candle fetching to the exchange-specific adapter."""
    exchange = symbol.exchange
    api_symbol = symbol.api_symbol
    kwargs = {"timestamp_from": timestamp_from, "timestamp_to": timestamp_to}
    if resolution is not None:
        kwargs["resolution"] = resolution
    if exchange in (Exchange.BINANCE, Exchange.BINANCE_FUTURES):
        kwargs["symbol_type"] = get_binance_symbol_type(symbol)
        candles = binance_candles(api_symbol, **kwargs)
    elif exchange == Exchange.BITFINEX:
        candles = bitfinex_candles(api_symbol, **kwargs)
    elif exchange == Exchange.BITMEX:
        candles = bitmex_candles(api_symbol, **kwargs)
    elif exchange == Exchange.COINBASE:
        candles = coinbase_candles(api_symbol, **kwargs)
    elif exchange == Exchange.COINBASE_ADVANCED:
        candles = coinbase_advanced_candles(api_symbol, **kwargs)
    elif exchange == Exchange.HYPERLIQUID:
        candles = hyperliquid_candles(api_symbol, **kwargs)
    else:
        raise NotImplementedError
    return candles


def funding_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Dispatch funding fetching to the exchange-specific adapter."""
    if symbol.symbol_type != SymbolType.PERPETUAL:
        raise ValueError("Funding is only available for perpetual symbols.")
    timestamp_range = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if timestamp_range is None:
        return DataFrame(columns=["timestamp"]).set_index("timestamp")
    timestamp_from, timestamp_to = timestamp_range
    exchange = symbol.exchange
    if exchange == Exchange.BINANCE_FUTURES:
        return binance_funding(symbol.api_symbol, timestamp_from, timestamp_to)
    if exchange == Exchange.BITMEX:
        return bitmex_funding(symbol.api_symbol, timestamp_from, timestamp_to)
    if exchange == Exchange.COINBASE_ADVANCED:
        return coinbase_advanced_funding(
            symbol.api_symbol,
            timestamp_from,
            timestamp_to,
        )
    if exchange == Exchange.HYPERLIQUID:
        return hyperliquid_funding(symbol.api_symbol, timestamp_from, timestamp_to)
    raise NotImplementedError(f"Funding is not implemented for {exchange}.")


def iter_funding_windows(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
):
    if symbol.exchange not in FUNDING_CHUNKED_EXCHANGES:
        yield timestamp_from, timestamp_to
        return
    yield from iter_chunks(
        timestamp_from,
        timestamp_to,
        value=FUNDING_FETCH_WINDOW,
        reverse=True,
    )


def funding(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
) -> None:
    """Fetch exchange funding and persist FundingData rows."""
    timestamp_range = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if timestamp_range is None:
        return
    timestamp_from, timestamp_to = timestamp_range
    for ts_from, ts_to in iter_funding_windows(symbol, timestamp_from, timestamp_to):
        fetch_timestamp_from = ts_from
        if not retry:
            if FundingData.objects.window_starts_after_range(symbol, ts_from, ts_to):
                return
            fetch_timestamp_from = FundingData.objects.next_fetch_timestamp_from(
                symbol,
                ts_from,
                ts_to,
            )
            if fetch_timestamp_from >= ts_to:
                continue

        data_frame = funding_api(symbol, fetch_timestamp_from, ts_to)
        FundingData.write(symbol, fetch_timestamp_from, ts_to, data_frame)
        if not retry and data_frame.empty:
            return


def exchange_candles_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = None,
) -> DataFrame:
    """Fetch direct exchange candles without writing trade-derived CandleData."""
    resolution = resolution or symbol.exchange_candle_resolution
    if not resolution:
        raise ValueError("Exchange candle resolution is required.")
    timestamp_range = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if timestamp_range is None:
        return DataFrame(columns=["timestamp"]).set_index("timestamp")
    timestamp_from, timestamp_to = timestamp_range
    return candles_api(
        symbol,
        timestamp_from,
        timestamp_to,
        resolution=resolution,
    )


def exchange_candles(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = None,
    retry: bool = False,
) -> None:
    """Fetch direct exchange candles and persist ExchangeCandleData rows."""
    resolution = resolution or symbol.exchange_candle_resolution
    if not resolution:
        raise ValueError("Exchange candle resolution is required.")
    frequency = parse_fixed_resolution_minutes(resolution)
    data_frame = exchange_candles_api(
        symbol,
        timestamp_from,
        timestamp_to,
        resolution=resolution,
    )
    if retry:
        ExchangeCandleData.objects.in_range(
            symbol,
            frequency,
            timestamp_from,
            timestamp_to,
        ).delete()
    ExchangeCandleData.write(
        symbol,
        frequency,
        timestamp_from,
        timestamp_to,
        data_frame,
    )
