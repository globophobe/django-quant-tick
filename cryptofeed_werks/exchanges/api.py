from datetime import datetime
from typing import Callable, Dict, Optional

from pandas import DataFrame

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.models import AggregatedTradeData, Symbol

from .binance import binance_trades
from .bitfinex import bitfinex_trades
from .bitmex import bitmex_trades
from .bybit import bybit_trades
from .coinbase import coinbase_trades
from .ftx import ftx_trades

# from .bitflyer import bitflyer_trades
# from .upbit import UPBIT, upbit_trades
# from .deribit import DERIBIT, deribit_trades


def api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
    verbose: bool = False,
):
    """API."""

    def on_data_frame(
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        validated: Optional[Dict[datetime, bool]] = {},
    ) -> DataFrame:
        """On data_frame, write aggregated data."""
        AggregatedTradeData.write(
            symbol,
            timestamp_from,
            timestamp_to,
            data_frame,
            validated=validated,
        )

    exchange_api(
        symbol=symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    )


def exchange_api(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
):
    """Exchange API."""
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
    # elif exchange == DERIBIT:
    #     deribit_trades(**kwargs)
    elif exchange == Exchange.FTX:
        ftx_trades(symbol, **kwargs)
    # elif exchange == UPBIT:
    #     upbit_trades(**kwargs)
