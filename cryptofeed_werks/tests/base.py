import random
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
from uuid import uuid4

import pandas as pd
from django.test import TestCase
from pandas import DataFrame


class RandomTradeTestCase(TestCase):
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
