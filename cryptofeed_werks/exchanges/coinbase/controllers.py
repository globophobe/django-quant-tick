from datetime import date, datetime
from typing import Callable, Optional

from pandas import DataFrame

from cryptofeed_werks.controllers import ExchangeREST
from cryptofeed_werks.models import Symbol

from .base import CoinbaseMixin
from .constants import BTCUSD


def coinbase_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
):
    CoinbaseTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class CoinbaseTrades(CoinbaseMixin, ExchangeREST):
    def assert_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        trades: Optional[list] = None,
    ) -> None:
        super().assert_data_frame(timestamp_from, timestamp_to, data_frame, trades)
        # Missing orders.
        expected = len(trades) - 1
        if self.symbol.name == BTCUSD:
            # It seems 45 ids may have been skipped for BTC-USD on 2021-06-09
            if timestamp_from.date() == date(2021, 6, 9):
                return
            # There was a missing order for BTC-USD on 2019-04-11
            elif timestamp_from.date() == date(2019, 4, 11):
                return
        diff = data_frame["index"].diff().dropna()
        assert abs(diff.sum()) == expected
