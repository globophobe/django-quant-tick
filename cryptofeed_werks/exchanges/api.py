from datetime import datetime
from typing import Callable

from cryptofeed_werks.constants import Exchange, SymbolType
from cryptofeed_werks.models import Symbol

from .binance import binance_trades
from .bitfinex import bitfinex_trades
from .bitflyer import bitflyer_trades
from .bitmex import bitmex_futures, bitmex_trades
from .bybit import bybit_trades
from .coinbase import coinbase_trades
from .ftx import BTCMOVE  # ftx_futures
from .ftx import ftx_move, ftx_trades

# from .upbit import UPBIT, upbit_trades
# from .deribit import DERIBIT, deribit_trades


def exchange_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
):
    exchange = symbol.exchange
    futures = symbol.symbol_type == SymbolType.FUTURE
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": on_data_frame,
        "retry": retry,
        "verbose": verbose,
    }
    if exchange == Exchange.BINANCE:
        if futures:
            raise NotImplementedError
        else:
            binance_trades(symbol, **kwargs)
    elif exchange == Exchange.BITFINEX:
        if futures:
            raise NotImplementedError
        else:
            bitfinex_trades(symbol, **kwargs)
    elif exchange == Exchange.BITFLYER:
        bitflyer_trades(symbol, **kwargs)
    elif exchange == Exchange.BITMEX:
        if futures:
            bitmex_futures(symbol, **kwargs)
        else:
            bitmex_trades(symbol, **kwargs)
    elif exchange == Exchange.BYBIT:
        if futures:
            raise NotImplementedError
        else:
            bybit_trades(symbol, **kwargs)
    elif exchange == Exchange.COINBASE:
        if futures:
            raise NotImplementedError
        else:
            coinbase_trades(symbol, **kwargs)
    # elif exchange == DERIBIT:
    #     deribit_trades(**kwargs)
    elif exchange == Exchange.FTX:
        if symbol == BTCMOVE:
            ftx_move(**kwargs)
        elif futures:
            raise NotImplementedError
        else:
            ftx_trades(symbol, **kwargs)
    # elif exchange == UPBIT:
    #     upbit_trades(**kwargs)
