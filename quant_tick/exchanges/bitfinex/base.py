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

    def parse_data(self, data: list) -> list:
        parsed = []
        for trade in data:
            timestamp = get_bitfinex_api_timestamp(trade)
            amount = Decimal(trade[2])
            price = Decimal(trade[3])
            notional = abs(amount)
            parsed.append(
                {
                    "uid": str(trade[0]),
                    "timestamp": timestamp,
                    "nanoseconds": 0,
                    "price": price,
                    "volume": price * notional,
                    "notional": notional,
                    "tickRule": np.sign(amount),
                    "index": trade[0],
                }
            )
        return parsed

    def get_data_frame(self, trades: list) -> list:
        """Build a DataFrame from unsorted Bitfinex REST trades.

        Websocket sends trades in order, by incrementing non sequential integer
        REST API returns results unsorted
        Sort by uid, reversed
        """
        return super().get_data_frame(trades)

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        return bitfinex_candles(
            self.symbol.api_symbol, timestamp_from, timestamp_to, time_frame="1m"
        )
