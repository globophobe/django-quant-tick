from datetime import datetime
from pathlib import Path

import pandas as pd
from django.core.files.storage import default_storage
from pandas import DataFrame

from quant_candles.lib import aggregate_trades, volume_filter_with_time_window
from quant_candles.models import Symbol, TradeData

from ..base import BaseSymbolTest, RandomTradeTest


class BaseWriteTradeDataTest(RandomTradeTest, BaseSymbolTest):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1t")

    def get_filtered(self, timestamp: datetime, nanoseconds: int = 0) -> DataFrame:
        """Get filtered."""
        trades = [self.get_random_trade(timestamp=timestamp, nanoseconds=nanoseconds)]
        data_frame = pd.DataFrame(trades)
        aggregated = aggregate_trades(data_frame)
        return volume_filter_with_time_window(aggregated, min_volume=None, window="1t")

    def tearDown(self):
        trade_data = TradeData.objects.select_related("symbol")
        # Files
        for obj in trade_data:
            obj.delete()
        # Directories
        symbols = Symbol.objects.all()
        trades = Path("trades")
        for obj in symbols:
            exchange = obj.exchange
            symbol = obj.symbol
            head = trades / exchange / symbol
            if obj.should_aggregate_trades:
                tail = Path("aggregated") / str(obj.significant_trade_filter)
            else:
                tail = "raw"
            path = head / tail
            directories, _ = default_storage.listdir(str(path.resolve()))
            for directory in directories:
                default_storage.delete(path / directory)
            default_storage.delete(path)
            if obj.should_aggregate_trades:
                default_storage.delete(trades / exchange / symbol / "aggregated")
            default_storage.delete(trades / exchange / symbol)
        for exchange in [symbol.exchange for symbol in symbols]:
            default_storage.delete(trades / exchange)
        default_storage.delete(trades)
