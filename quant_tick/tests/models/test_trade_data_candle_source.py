from __future__ import annotations

from decimal import Decimal

import pandas as pd
from django.test import TestCase

from quant_tick.lib import (
    aggregate_candle,
    aggregate_trades,
    get_min_time,
    get_next_time,
    merge_cache,
    volume_filter_with_time_window,
)
from quant_tick.models import TradeData
from quant_tick.storage import convert_trade_data_to_daily

from ..base import BaseWriteTradeDataTest


class TradeDataCandleSourceTests(BaseWriteTradeDataTest, TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1min")

    def test_write_trade_candle_uses_aggregated_dataset_when_available(self):
        symbol = self.get_symbol(save_raw=True, save_aggregated=True)
        raw = pd.DataFrame(
            [
                self.get_random_trade(
                    timestamp=self.timestamp_from,
                    nanoseconds=0,
                    price=Decimal("1"),
                    notional=Decimal("1"),
                    tick_rule=1,
                ),
                self.get_random_trade(
                    timestamp=self.timestamp_from,
                    nanoseconds=0,
                    price=Decimal("3"),
                    notional=Decimal("1"),
                    tick_rule=1,
                ),
            ]
        )

        TradeData.write(
            symbol, self.timestamp_from, self.timestamp_to, raw, pd.DataFrame([])
        )

        trade_data = TradeData.objects.get()
        aggregated = aggregate_trades(raw)
        self.assertEqual(
            trade_data.json_data["candle"],
            aggregate_candle(
                aggregated,
                min_volume_exponent=1,
                min_notional_exponent=1,
            ),
        )

    def test_write_trade_candle_uses_filtered_dataset_when_available(self):
        symbol = self.get_symbol(
            save_raw=True,
            save_aggregated=True,
            significant_trade_filter=2,
        )
        raw = pd.DataFrame(
            [
                self.get_random_trade(
                    timestamp=self.timestamp_from,
                    nanoseconds=0,
                    price=Decimal("1"),
                    notional=Decimal("1"),
                    tick_rule=1,
                ),
                self.get_random_trade(
                    timestamp=self.timestamp_from + pd.Timedelta("10s"),
                    nanoseconds=0,
                    price=Decimal("3"),
                    notional=Decimal("1"),
                    tick_rule=1,
                ),
            ]
        )

        TradeData.write(
            symbol, self.timestamp_from, self.timestamp_to, raw, pd.DataFrame([])
        )

        trade_data = TradeData.objects.get()
        filtered = volume_filter_with_time_window(
            aggregate_trades(raw),
            min_volume=symbol.significant_trade_filter,
        )
        self.assertEqual(
            trade_data.json_data["candle"],
            aggregate_candle(
                filtered,
                min_volume_exponent=1,
                min_notional_exponent=1,
            ),
        )

    def test_convert_trade_data_to_daily_preserves_full_candle_payload(self):
        symbol = self.get_symbol(
            save_raw=True,
            save_aggregated=True,
            significant_trade_filter=2,
        )
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}min")
            ts_to = ts_from + pd.Timedelta("1min")
            raw = pd.DataFrame(
                [
                    self.get_random_trade(
                        timestamp=ts_from,
                        nanoseconds=0,
                        price=Decimal("1000"),
                        notional=Decimal("1"),
                        tick_rule=1,
                    )
                ]
            )
            TradeData.write(symbol, ts_from, ts_to, raw, pd.DataFrame([]))

        candles = [
            dict(item["json_data"]["candle"])
            for item in TradeData.objects.filter(symbol=symbol)
            .order_by("timestamp")
            .values("json_data")
        ]
        expected = candles[0]
        for candle in candles[1:]:
            expected = merge_cache(expected, dict(candle))

        convert_trade_data_to_daily(
            symbol, timestamp_from, get_next_time(timestamp_from, value="1h")
        )

        trade_data = TradeData.objects.get(symbol=symbol)
        self.assertEqual(trade_data.json_data["candle"], expected)
