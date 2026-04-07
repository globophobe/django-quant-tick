from datetime import UTC, datetime
from decimal import Decimal

from django.test import SimpleTestCase

from quant_tick.exchanges.bitmex.candles import dedupe_candles


class BitmexCandleTest(SimpleTestCase):

    def test_dedupe_candles_removes_identical_duplicates(self):
        ts = datetime(2015, 10, 20, tzinfo=UTC)
        candle = {
            "timestamp": ts,
            "open": Decimal("1"),
            "high": Decimal("1"),
            "low": Decimal("1"),
            "close": Decimal("1"),
            "volume": Decimal("1"),
        }
        next_candle = candle.copy()
        next_candle["timestamp"] = datetime(2015, 10, 20, 0, 1, tzinfo=UTC)

        candles = dedupe_candles([candle, candle.copy(), next_candle])

        self.assertEqual(candles, [candle, next_candle])

    def test_dedupe_candles_rejects_conflicting_duplicates(self):
        ts = datetime(2015, 10, 20, tzinfo=UTC)
        candle = {"timestamp": ts, "volume": Decimal("1")}
        duplicate = {"timestamp": ts, "volume": Decimal("2")}

        with self.assertRaises(ValueError):
            dedupe_candles([candle, duplicate])
