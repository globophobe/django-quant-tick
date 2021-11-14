from datetime import datetime
from decimal import Decimal
from typing import Optional

import numpy as np

from cryptofeed_werks.models import Symbol

from .api import get_bitfinex_api_timestamp, get_trades


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

    def get_pagination_id(self, data: Optional[dict] = None) -> int:
        """Get pagination_id."""
        return int(self.timestamp_to.timestamp() * 1000)  # Millisecond

    def iter_api(self, symbol: Symbol, pagination_id: int) -> list:
        """Iterate Bitfinex API."""
        return get_trades(symbol, self.timestamp_from, pagination_id, self.log_format)

    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        return str(trade[0])

    def get_timestamp(self, trade: dict) -> datetime:
        """Get timestamp."""
        return get_bitfinex_api_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        """Get nanoseconds."""
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        """Get price."""
        return Decimal(trade[3])

    def get_volume(self, trade: dict) -> Decimal:
        """Get volume."""
        return self.get_price(trade) * self.get_notional(trade)

    def get_notional(self, trade: dict) -> Decimal:
        """Get notional."""
        return abs(Decimal(trade[2]))

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule.

        Buy side indicates a down-tick because the maker was a buy order and
        their order was removed. Conversely, sell side indicates an up-tick.
        """
        return np.sign(trade[2])

    def get_index(self, trade: dict) -> int:
        """Get index."""
        return trade[0]

    def get_data_frame(self, trades: list) -> list:
        """Get data_frame.

        Websocket sends trades in order, by incrementing non sequential integer
        REST API returns results unsorted
        Sort by uid, reversed
        """
        trades.sort(key=lambda x: x["index"], reverse=True)
        return super().get_data_frame(trades)
