from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange
from quant_tick.exchanges.api import candles_api
from quant_tick.exchanges.bitmex.candles import (
    bitmex_candles,
    dedupe_candles,
    get_bitmex_fetch_bin_size,
    resample_bitmex_candles,
)


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

    def test_get_bitmex_fetch_bin_size_uses_hourly_fetch_for_2h(self):
        target_minutes, fetch_bin_size = get_bitmex_fetch_bin_size("2h")

        self.assertEqual(target_minutes, 120)
        self.assertEqual(fetch_bin_size, "1h")

    def test_resample_bitmex_candles_aggregates_hourly_rows_to_2h(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 2, tzinfo=UTC)
        data_frame = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + pd.Timedelta(f"{hour}h"),
                    "open": Decimal(str(hour + 1)),
                    "high": Decimal(str(hour + 11)),
                    "low": Decimal(str(hour)),
                    "close": Decimal(str(hour + 2)),
                    "volume": Decimal("10"),
                }
                for hour in range(2)
            ]
        ).set_index("timestamp")

        candles = resample_bitmex_candles(
            data_frame,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution_minutes=120,
        )

        self.assertEqual(list(candles.index), [timestamp_from])
        candle = candles.iloc[0]
        self.assertEqual(candle.open, Decimal("1"))
        self.assertEqual(candle.high, Decimal("12"))
        self.assertEqual(candle.low, Decimal("0"))
        self.assertEqual(candle.close, Decimal("3"))
        self.assertEqual(candle.volume, Decimal("20"))

    def test_bitmex_candles_resamples_to_requested_2h_resolution(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 2, tzinfo=UTC)
        source_candles = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + pd.Timedelta(f"{hour}h"),
                    "open": Decimal(str(hour + 1)),
                    "high": Decimal(str(hour + 11)),
                    "low": Decimal(str(hour)),
                    "close": Decimal(str(hour + 2)),
                    "volume": Decimal("10"),
                }
                for hour in range(2)
            ]
        ).set_index("timestamp")

        with patch(
            "quant_tick.exchanges.bitmex.candles.fetch_bitmex_candles",
            return_value=source_candles,
        ) as fetch_candles:
            result = bitmex_candles(
                "XBTUSD",
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        fetch_candles.assert_called_once_with(
            "XBTUSD",
            timestamp_from,
            timestamp_to,
            bin_size="1h",
            log_format=None,
        )
        self.assertEqual(list(result.index), [timestamp_from])
        candle = result.iloc[0]
        self.assertEqual(candle.open, Decimal("1"))
        self.assertEqual(candle.high, Decimal("12"))
        self.assertEqual(candle.low, Decimal("0"))
        self.assertEqual(candle.close, Decimal("3"))
        self.assertEqual(candle.volume, Decimal("20"))

    def test_candles_api_passes_resolution_to_bitmex(self):
        symbol = SimpleNamespace(exchange=Exchange.BITMEX, api_symbol="XBTUSD")
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 2, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.bitmex_candles",
            return_value=expected,
        ) as mocked:
            result = candles_api(
                symbol,
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        mocked.assert_called_once_with(
            "XBTUSD",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="2h",
        )
        self.assertTrue(result.equals(expected))
