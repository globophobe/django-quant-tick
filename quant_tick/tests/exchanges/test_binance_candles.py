from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange
from quant_tick.exchanges.api import candles_api
from quant_tick.exchanges.binance.candles import (
    binance_candles,
    get_binance_interval,
)


class BinanceCandleTest(SimpleTestCase):

    def test_get_binance_interval_has_2h(self):
        interval = get_binance_interval("2h")

        self.assertEqual(interval, "2h")

    def test_get_binance_interval_has_1M(self):
        interval = get_binance_interval("1M")

        self.assertEqual(interval, "1M")

    def test_binance_candles_uses_requested_2h_interval(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 4, tzinfo=UTC)
        candles = [
            [
                int((timestamp_from + pd.Timedelta("2h")).timestamp() * 1000),
                "1.5",
                "3",
                "1",
                "2",
                "12",
            ],
            [
                int(timestamp_from.timestamp() * 1000),
                "1",
                "2",
                "0",
                "1.5",
                "10",
            ],
        ]

        with patch(
            "quant_tick.exchanges.binance.candles.iter_api",
            return_value=(candles, True, None),
        ) as mocked:
            result = binance_candles(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        self.assertEqual(list(result.index), [timestamp_from, timestamp_from + pd.Timedelta("2h")])
        self.assertEqual(result.iloc[0].open, Decimal("1"))
        self.assertEqual(result.iloc[1].close, Decimal("2"))
        url = mocked.call_args.args[0]
        self.assertIn("interval=2h", url)
        self.assertEqual(
            mocked.call_args.kwargs["pagination_id"],
            int((timestamp_to - pd.Timedelta("2h")).timestamp() * 1000),
        )

    def test_candles_api_passes_resolution_to_binance(self):
        symbol = SimpleNamespace(exchange=Exchange.BINANCE, api_symbol="BTCUSDT")
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 2, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.binance_candles",
            return_value=expected,
        ) as mocked:
            result = candles_api(
                symbol,
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        mocked.assert_called_once_with(
            "BTCUSDT",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="2h",
        )
        self.assertTrue(result.equals(expected))
