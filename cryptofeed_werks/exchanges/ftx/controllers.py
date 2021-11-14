import re
from datetime import datetime
from typing import Callable

from cryptofeed_werks.controllers import ExchangeMultiSymbolREST, ExchangeREST
from cryptofeed_werks.models import Symbol

from .base import FTXMixin
from .constants import BTC, BTCMOVE
from .trades import get_active_futures, get_expired_futures


def ftx_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
):
    FTX(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


def ftx_move(
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
):
    FTXMOVE(
        api_symbol=BTC,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class FTX(FTXMixin, ExchangeREST):
    pass


class FTXMOVE(FTXMixin, ExchangeMultiSymbolREST):
    @property
    def log_format(self):
        symbol = BTCMOVE.replace("-", "")
        return f"{self.exchange_display} {symbol}"

    def get_suffix(self, sep="-"):
        return BTCMOVE.replace("-", "")

    def get_symbols(self):
        active_futures = get_active_futures(self.symbol, verbose=False)
        expired_futures = get_expired_futures(self.symbol, verbose=False)
        futures = active_futures + expired_futures
        regex = re.compile(r"^BTC-MOVE-(WK)?-?(\d{4})?(\d{4})?(Q\d)?$")
        move_futures = []
        for future in futures:
            api_symbol = future["api_symbol"]
            match = regex.match(api_symbol)
            if match:
                week = match.group(1)
                period = match.group(3) or match.group(2)
                quarter = match.group(4)
                if week:
                    future["symbol"] = f"{week}{period}"
                if quarter:
                    future["symbol"] = f"{period}{quarter}"
                if not week and not quarter:
                    future["symbol"] = f"D{period}"
                move_futures.append(future)
            elif api_symbol == "BTC-MOVE-20202020Q1":
                future["symbol"] = "2020Q1"
                move_futures.append(future)
        return move_futures
