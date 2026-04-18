from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange
from quant_tick.exchanges.api import candles_api
from quant_tick.exchanges.bitfinex.candles import (
    bitfinex_candles,
    get_bitfinex_fetch_time_frame,
)


class BitfinexCandleTest(SimpleTestCase):

    def test_get_bitfinex_fetch_time_frame_uses_hourly_fetch_for_2h(self):
        target_minutes, fetch_time_frame = get_bitfinex_fetch_time_frame("2h")

        self.assertEqual(target_minutes, 120)
        self.assertEqual(fetch_time_frame, "1h")

    def test_get_bitfinex_fetch_time_frame_keeps_native_12h(self):
        target_minutes, fetch_time_frame = get_bitfinex_fetch_time_frame("12h")

        self.assertEqual(target_minutes, 720)
        self.assertEqual(fetch_time_frame, "12h")

    def test_bitfinex_candles_resamples_to_requested_2h_resolution(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 2, tzinfo=UTC)
        native = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + pd.Timedelta(f"{hour}h"),
                    "open": Decimal("1"),
                    "high": Decimal("2"),
                    "low": Decimal("0"),
                    "close": Decimal("1"),
                    "notional": Decimal("1"),
                }
                for hour in range(2)
            ]
        ).set_index("timestamp")
        expected = native.iloc[:1]

        with (
            patch(
                "quant_tick.exchanges.bitfinex.candles.fetch_bitfinex_candles",
                return_value=native,
            ) as fetch,
            patch(
                "quant_tick.exchanges.bitfinex.candles.resample_candles",
                return_value=expected,
            ) as resample,
        ):
            result = bitfinex_candles(
                "tBTCUSD",
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        fetch.assert_called_once_with(
            "tBTCUSD",
            timestamp_from,
            timestamp_to,
            time_frame="1h",
            log_format=None,
        )
        resample.assert_called_once_with(
            native,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution_minutes=120,
        )
        self.assertTrue(result.equals(expected))

    def test_candles_api_passes_resolution_to_bitfinex(self):
        symbol = SimpleNamespace(exchange=Exchange.BITFINEX, api_symbol="tBTCUSD")
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 2, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.bitfinex_candles",
            return_value=expected,
        ) as mocked:
            result = candles_api(
                symbol,
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        mocked.assert_called_once_with(
            "tBTCUSD",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="2h",
        )
        self.assertTrue(result.equals(expected))
