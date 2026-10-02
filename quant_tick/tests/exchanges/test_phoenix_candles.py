from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import exchange_candles
from quant_tick.exchanges.phoenix.candles import phoenix_candles
from quant_tick.models import ExchangeCandleData, Symbol


def candle(timestamp, offset=0):
    return {
        "time": int(timestamp.timestamp() * 1000),
        "open": 100 + offset,
        "high": 102 + offset,
        "low": 99 + offset,
        "close": 101 + offset,
        "markOpen": 200 + offset,
        "markHigh": 202 + offset,
        "markLow": 199 + offset,
        "markClose": 201 + offset,
        "volume": "0.1",
        "volumeQuote": "10.1",
        "tradeCount": 2,
    }


class PhoenixCandleTest(SimpleTestCase):
    def setUp(self):
        self.start = datetime(2026, 9, 23, tzinfo=UTC)

    def test_only_complete_eight_hour_buckets_are_returned(self):
        for hours in ((0,), (4,), (0, 4)):
            with (
                self.subTest(hours=hours),
                patch(
                    "quant_tick.exchanges.phoenix.candles.get_phoenix_response",
                    return_value=[
                        candle(self.start + timedelta(hours=hour), hour)
                        for hour in hours
                    ],
                ) as response,
            ):
                result = phoenix_candles(
                    "btc", self.start, self.start + timedelta(hours=8), "8h"
                )
            self.assertEqual(response.call_args.args[1]["timeframe"], "4h")
            if len(hours) == 1:
                self.assertTrue(result.empty)
                continue
            self.assertEqual(list(result.index), [self.start])
            row = result.iloc[0]
            self.assertEqual(
                (row.open, row.high, row.low, row.close), (100, 106, 99, 105)
            )
            self.assertEqual(
                (row.mark_open, row.mark_high, row.mark_low, row.mark_close),
                (200, 206, 199, 205),
            )
            self.assertEqual(row.notional, Decimal("0.2"))
            self.assertEqual(row.volume, Decimal("20.2"))
            self.assertEqual(row.trades, 4)

    def test_chunks_respect_inclusive_api_end_and_discard_unfinished_tail(self):
        end = self.start + timedelta(minutes=3, seconds=30)
        with (
            patch("quant_tick.exchanges.phoenix.candles.CANDLE_MAX_RESULTS", 3),
            patch(
                "quant_tick.exchanges.phoenix.candles.get_phoenix_response",
                side_effect=[
                    [
                        candle(self.start + timedelta(minutes=minute))
                        for minute in (0, 1)
                    ],
                    [
                        candle(self.start + timedelta(minutes=minute))
                        for minute in (2, 3)
                    ],
                ],
            ) as response,
        ):
            result = phoenix_candles("BTC", self.start, end)
        self.assertEqual(len(result), 3)
        params = [item.args[1] for item in response.call_args_list]
        self.assertEqual(params[0]["endTime"] + 1, params[1]["startTime"])
        self.assertEqual(params[1]["endTime"], int(end.timestamp() * 1000) - 1)
        self.assertTrue(all(item["enableExternalSource"] == "false" for item in params))

    def test_external_candles_are_rejected(self):
        with (
            patch(
                "quant_tick.exchanges.phoenix.candles.get_phoenix_response",
                return_value=[candle(self.start) | {"externalSource": "other-venue"}],
            ),
            self.assertRaisesRegex(ValueError, "external-source"),
        ):
            phoenix_candles("BTC", self.start, self.start + timedelta(minutes=1))


class PhoenixCandleCollectionTest(TestCase):
    def test_dispatch_persists_units_and_mark_prices(self):
        start = datetime(2026, 9, 23, tzinfo=UTC)
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="4h",
        )
        with patch(
            "quant_tick.exchanges.phoenix.candles.get_phoenix_response",
            return_value=[candle(start)],
        ):
            exchange_candles(symbol, start, start + timedelta(hours=4))
        stored = ExchangeCandleData.objects.get(symbol=symbol)
        self.assertEqual(stored.frequency, 240)
        self.assertEqual(stored.notional, Decimal("0.1"))
        self.assertEqual(stored.volume, Decimal("10.1"))
        self.assertEqual(Decimal(stored.json_data["mark_close"]), Decimal(201))
        self.assertEqual(stored.json_data["trades"], 2)
