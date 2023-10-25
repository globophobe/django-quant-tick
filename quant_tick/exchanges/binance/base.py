from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.controllers import SequentialIntegerMixin

from .api import get_binance_api_timestamp, get_trades


class BinanceMixin(SequentialIntegerMixin):
    def iter_api(self, timestamp_from: datetime, pagination_id: str) -> tuple:
        """Iterate Binance API."""
        return get_trades(
            self.symbol.api_symbol, timestamp_from, pagination_id, self.log_format
        )

    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        return str(trade["id"])

    def get_timestamp(self, trade: dict) -> datetime:
        """Get timestamp."""
        return get_binance_api_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        """Get nanoseconds."""
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        """Get price."""
        return Decimal(trade["price"])

    def get_volume(self, trade: dict) -> Decimal:
        """Get volume."""
        return self.get_price(trade) * self.get_notional(trade)

    def get_notional(self, trade: dict) -> Decimal:
        """Get notional."""
        return Decimal(trade["qty"])

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule.

        If isBuyerMaker is true, order was filled by sell order
        """
        return 1 if not trade["isBuyerMaker"] else -1

    def get_index(self, trade: dict) -> int:
        """Get index."""
        return int(trade["id"])

    def assert_data_frame(self, data_frame: DataFrame, trades: list) -> None:
        """Assertions on data_frame."""
        # Duplicates.
        assert len(data_frame["uid"].unique()) == len(trades)
        # Missing orders.
        expected = len(trades) - 1
        diff = data_frame["index"].diff().dropna()
        assert abs(diff.sum()) == expected

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get candles from Exchange API."""
        return pd.DataFrame([])
