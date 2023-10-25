import random
from datetime import datetime, timezone
from decimal import Decimal
from itertools import chain
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import pandas as pd
from django.core.files.storage import default_storage
from pandas import DataFrame

from quant_tick.constants import Exchange
from quant_tick.lib import (
    aggregate_trades,
    get_current_time,
    get_min_time,
    volume_filter_with_time_window,
)
from quant_tick.models import GlobalSymbol, Symbol, TradeData, TradeDataSummary


class BaseRandomTradeTest:
    def generate_random_trades(
        self,
        ticks: int,
        symbol: Optional[str] = None,
        prices: list = [],
        is_equal_timestamp: bool = False,
        nanoseconds: int = 0,
        notional: Optional[Decimal] = None,
        total_ticks: int = 1,
    ):
        """Generate random trades"""
        trades = []
        for index, tick in enumerate(ticks):
            # Price
            if index == 0:
                if len(prices):
                    price = self.get_next_price(prices[index], tick)
                else:
                    price = None
            else:
                price = self.get_next_price(trades[-1]["price"], tick, jitter=0.1)
            # Timestamp
            if len(trades) and is_equal_timestamp:
                timestamp = trades[0]["timestamp"]
            else:
                timestamp = None
            trades.append(
                self.get_random_trade(
                    symbol=symbol,
                    timestamp=timestamp,
                    nanoseconds=nanoseconds,
                    price=price,
                    notional=notional,
                    tick_rule=tick,
                    total_ticks=total_ticks,
                )
            )
        return trades

    def get_next_price(
        self, price: Decimal, tick_rule: int, jitter: float = 0.0
    ) -> Decimal:
        """Get next price."""
        change = random.random() * tick_rule * jitter
        return price + Decimal(str(round(change, 2)))  # Change in dollars

    def get_random_trade(
        self,
        symbol: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        nanoseconds: int = 0,
        price: Optional[Decimal] = None,
        notional: Optional[Decimal] = None,
        tick_rule: Optional[int] = None,
        total_ticks: int = 1,
    ) -> dict:
        """Get random trade."""
        timestamp = timestamp or datetime.now()
        price = price or Decimal(str(round(random.random() * 10, 2)))
        notional = notional or Decimal(str(random.random() * 10))
        volume = price * notional
        tick_rule = tick_rule or random.choice((1, -1))
        data = {
            "uid": uuid4().hex,
            "timestamp": timestamp.replace(tzinfo=timezone.utc),
            "nanoseconds": nanoseconds,
            "price": price,
            "volume": volume,
            "notional": notional,
            "tickRule": tick_rule,
            "ticks": total_ticks,
        }
        if symbol:
            data["symbol"] = symbol
        return data

    def get_data_frame(self, data: List[dict]) -> DataFrame:
        """Get data_frame."""
        trades = []
        for item in data:
            prices = item.pop("prices", [])
            ticks = item.pop("ticks")
            trades += self.generate_random_trades(ticks, prices=prices, **item)
        return pd.DataFrame(trades)


class BaseSymbolTest:
    def setUp(self):
        self.global_symbol = GlobalSymbol.objects.create(name="global-symbol")
        self.timestamp_from = get_min_time(get_current_time(), "1d")

    def get_symbol(
        self, api_symbol: str = "test", should_aggregate_trades: bool = True
    ) -> Symbol:
        """Get symbol."""
        return Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.COINBASE,
            api_symbol=api_symbol,
            should_aggregate_trades=should_aggregate_trades,
        )


class BaseWriteTradeDataTest(BaseRandomTradeTest, BaseSymbolTest):
    def get_filtered(
        self,
        timestamp: datetime,
        nanoseconds: int = 0,
        price: Optional[Decimal] = None,
        notional: Optional[Decimal] = None,
    ) -> DataFrame:
        """Get filtered."""
        trades = [
            self.get_random_trade(
                timestamp=timestamp,
                nanoseconds=nanoseconds,
                price=price,
                notional=notional,
            )
        ]
        data_frame = pd.DataFrame(trades)
        aggregated = aggregate_trades(data_frame)
        return volume_filter_with_time_window(aggregated, min_volume=None, window="1t")

    def tearDown(self):
        trade_data_summary = TradeDataSummary.objects.all()
        trade_data = TradeData.objects.select_related("symbol")
        # Files
        for obj in list(chain(trade_data_summary, trade_data)):
            obj.delete()
        # Directories
        symbols = Symbol.objects.all()
        trades = Path("trades")
        for obj in symbols:
            exchange = obj.exchange
            symbol = obj.symbol
            path = trades / exchange / symbol
            raw_paths = [path / "raw" / "summary", path / "raw" / "data", path / "raw"]
            aggregated_paths = [
                path / "aggregated" / str(obj.significant_trade_filter) / "data",
                path / "aggregated" / str(obj.significant_trade_filter) / "summary",
                path / "aggregated" / str(obj.significant_trade_filter),
                path / "aggregated",
            ]
            for p in raw_paths + aggregated_paths:
                if p.exists():
                    directories, _ = default_storage.listdir(str(p.resolve()))
                    for directory in directories:
                        default_storage.delete(p / directory)
                    default_storage.delete(p)
            default_storage.delete(path)
        for exchange in [obj.exchange for obj in symbols]:
            default_storage.delete(trades / exchange)
        default_storage.delete(trades)
