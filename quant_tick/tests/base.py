import random
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pandas as pd
from django.core.files.storage import default_storage
from pandas import DataFrame

from quant_tick.constants import Exchange, SymbolType
from quant_tick.lib import (
    aggregate_trades,
    get_current_time,
    get_min_time,
    volume_filter_with_time_window,
)
from quant_tick.models import Symbol, TradeData


class BaseRandomTradeTest:
    def generate_random_trades(
        self,
        ticks: list[int],
        symbol: str | None = None,
        prices: list | None = None,
        is_equal_timestamp: bool = False,
        nanoseconds: int = 0,
        notional: Decimal | None = None,
        total_ticks: int = 1,
    ):
        prices = prices or []
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
        change = random.random() * tick_rule * jitter
        return price + Decimal(str(round(change, 2)))  # Change in dollars

    def get_random_trade(
        self,
        symbol: str | None = None,
        timestamp: datetime | None = None,
        nanoseconds: int = 0,
        price: Decimal | None = None,
        notional: Decimal | None = None,
        tick_rule: int | None = None,
        total_ticks: int = 1,
    ) -> dict:
        timestamp = timestamp or datetime.now()
        price = price or Decimal(str(round(random.random() * 10, 2)))
        notional = notional or Decimal(str(random.random() * 10))
        volume = price * notional
        tick_rule = tick_rule or random.choice((1, -1))
        data = {
            "uid": uuid4().hex,
            "timestamp": timestamp.replace(tzinfo=UTC),
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

    def get_data_frame(self, data: list[dict]) -> DataFrame:
        trades = []
        for item in data:
            prices = item.pop("prices", [])
            ticks = item.pop("ticks")
            trades += self.generate_random_trades(ticks, prices=prices, **item)
        return pd.DataFrame(trades)


class BaseSymbolTest:
    def setUp(self):
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")

    def get_symbol(
        self,
        exchange: str = Exchange.COINBASE,
        api_symbol: str = "test",
        save_raw: bool = True,
        save_aggregated: bool = False,
        significant_trade_filter: int = 0,
        symbol_type: str = SymbolType.SPOT,
    ) -> Symbol:
        return Symbol.objects.create(
            exchange=exchange,
            api_symbol=api_symbol,
            symbol_type=symbol_type,
            save_raw=save_raw,
            save_aggregated=save_aggregated,
            significant_trade_filter=significant_trade_filter,
        )


class BaseWriteTradeDataTest(BaseRandomTradeTest, BaseSymbolTest):

    def get_raw(
        self,
        timestamp: datetime,
        nanoseconds: int = 0,
        price: Decimal | None = None,
        notional: Decimal | None = None,
        tick_rule: int | None = None,
    ) -> DataFrame:
        trades = [
            self.get_random_trade(
                timestamp=timestamp,
                nanoseconds=nanoseconds,
                price=price,
                notional=notional,
                tick_rule=tick_rule,
            )
        ]
        return pd.DataFrame(trades)

    def get_aggregated(
        self,
        timestamp: datetime,
        nanoseconds: int = 0,
        price: Decimal | None = None,
        notional: Decimal | None = None,
        tick_rule: int | None = None,
    ) -> DataFrame:
        data_frame = self.get_raw(timestamp, nanoseconds, price, notional, tick_rule)
        return aggregate_trades(data_frame)

    def get_filtered(
        self,
        timestamp: datetime,
        nanoseconds: int = 0,
        price: Decimal | None = None,
        notional: Decimal | None = None,
        min_volume: Decimal | None = None,
        tick_rule: int | None = None,
    ) -> DataFrame:
        data_frame = self.get_aggregated(
            timestamp, nanoseconds, price, notional, tick_rule
        )
        return volume_filter_with_time_window(
            data_frame, min_volume=min_volume, window="1min"
        )

    def tearDown(self):
        # Files
        for obj in TradeData.objects.all():
            obj.delete()
        # Directories
        test_path = Path("test-trades")
        for obj in Symbol.objects.all():
            upload_path = Path("/".join(obj.upload_path))
            for file_data in ("raw", "aggregated", "filtered"):
                file_path = Path(file_data)
                p = test_path / upload_path / file_path
                if p.exists():
                    directories, _ = default_storage.listdir(str(p.resolve()))
                    for directory in directories:
                        default_storage.delete(p / directory)
                    default_storage.delete(p)

        for obj in Symbol.objects.all():
            for index in reversed(range(2)):
                idx = index + 2
                p = test_path / Path("/".join(obj.upload_path[:idx]))
                if p.exists():
                    default_storage.delete(p)

        for obj in Symbol.objects.all():
            p = test_path / obj.exchange
            if p.exists():
                default_storage.delete(p)

        default_storage.delete(test_path)
