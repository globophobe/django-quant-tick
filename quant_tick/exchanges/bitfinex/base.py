from datetime import datetime
from decimal import Decimal

import numpy as np
from pandas import DataFrame

from .api import format_bitfinex_api_timestamp, get_bitfinex_api_timestamp
from .candles import bitfinex_candles
from .trades import get_trades


class BitfinexMixin:
    """
    Details: https://docs.bitfinex.com/reference#rest-public-trades

    ID	int	ID of the trade
    MTS	int	millisecond time stamp
    ±AMOUNT	float	How much was bought (positive) or sold (negative).
    PRICE	float	Price at which the trade was executed (trading tickers only)
    RATE	float	Rate at which funding transaction occurred (funding tickers only)
    PERIOD	int	Amount of time the funding transaction was for (funding tickers only)
    """

    def get_pagination_id(self, timestamp_from: datetime) -> int | None:
        return format_bitfinex_api_timestamp(timestamp_from)

    def iter_api(self, timestamp_from: datetime, pagination_id: int) -> list:
        return get_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            log_format=self.log_format,
        )

    def get_uid(self, trade: dict) -> str:
        return str(trade[0])

    def get_timestamp(self, trade: dict) -> datetime:
        return get_bitfinex_api_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        return Decimal(trade[3])

    def get_volume(self, trade: dict) -> Decimal:
        return self.get_price(trade) * self.get_notional(trade)

    def get_notional(self, trade: dict) -> Decimal:
        return abs(Decimal(trade[2]))

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule.

        Buy side indicates a down-tick because the maker was a buy order and
        their order was removed. Conversely, sell side indicates an up-tick.
        """
        return np.sign(trade[2])

    def get_index(self, trade: dict) -> int:
        return trade[0]

    def get_data_frame(self, trades: list) -> list:
        """Build a DataFrame from unsorted Bitfinex REST trades.

        Websocket sends trades in order, by incrementing non sequential integer
        REST API returns results unsorted
        Sort by uid, reversed
        """
        # TODO: Reverify against candles.
        return super().get_data_frame(trades)

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        return bitfinex_candles(
            self.symbol.api_symbol, timestamp_from, timestamp_to, time_frame="1m"
        )
