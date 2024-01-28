from collections.abc import Callable
from datetime import datetime

from pandas import DataFrame

from quant_tick.constants import Exchange
from quant_tick.models import Symbol, TradeData

from .binance import binance_candles, binance_trades
from .bitfinex import bitfinex_candles, bitfinex_trades
from .bitmex import bitmex_candles, bitmex_trades
from .bybit import bybit_candles, bybit_trades
from .coinbase import coinbase_candles, coinbase_trades

# from .bitflyer import bitflyer_trades, bitflyer_candles
# from .upbit import UPBIT, upbit_trades, upbit_candles


def api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
    verbose: bool = False,
) -> None:
    """API."""

    def on_data_frame(
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        trades: DataFrame,
        candles: DataFrame,
    ) -> DataFrame:
        """On data_frame, write trade data."""
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
    """Trades API."""
    exchange = symbol.exchange
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": on_data_frame,
        "retry": retry,
        "verbose": verbose,
    }
    if exchange == Exchange.BINANCE:
        binance_trades(symbol, **kwargs)
    elif exchange == Exchange.BITFINEX:
        bitfinex_trades(symbol, **kwargs)
    # elif exchange == Exchange.BITFLYER:
    #     bitflyer_trades(symbol, **kwargs)
    elif exchange == Exchange.BITMEX:
        bitmex_trades(symbol, **kwargs)
    elif exchange == Exchange.BYBIT:
        bybit_trades(symbol, **kwargs)
    elif exchange == Exchange.COINBASE:
        coinbase_trades(symbol, **kwargs)
    # elif exchange == UPBIT:
    #     upbit_trades(**kwargs)


def candles_api(
    symbol: Symbol, timestamp_from: datetime, timestamp_to: datetime
) -> DataFrame:
    """Candles API."""
    exchange = symbol.exchange
    api_symbol = symbol.api_symbol
    kwargs = {"timestamp_from": timestamp_from, "timestamp_to": timestamp_to}
    if exchange == Exchange.BINANCE:
        candles = binance_candles(api_symbol, **kwargs)
    elif exchange == Exchange.BITFINEX:
        candles = bitfinex_candles(api_symbol, **kwargs)
    # elif exchange == Exchange.BITFLYER:
    #     bitflyer_trades(symbol, **kwargs)
    elif exchange == Exchange.BITMEX:
        candles = bitmex_candles(api_symbol, **kwargs)
    elif exchange == Exchange.BYBIT:
        candles = bybit_candles(api_symbol, **kwargs)
    elif exchange == Exchange.COINBASE:
        candles = coinbase_candles(api_symbol, **kwargs)
    # elif exchange == UPBIT:
    #     upbit_trades(**kwargs)
    else:
        raise NotImplementedError
    return candles
