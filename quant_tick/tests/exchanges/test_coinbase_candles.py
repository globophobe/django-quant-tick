from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange
from quant_tick.exchanges.api import candles_api
from quant_tick.exchanges.coinbase.candles import (
    coinbase_candles,
    fetch_coinbase_candles,
    get_coinbase_fetch_granularity,
)


class CoinbaseCandleTest(SimpleTestCase):

    def test_get_coinbase_fetch_granularity_uses_hourly_fetch_for_2h(self):
        target_minutes, fetch_granularity = get_coinbase_fetch_granularity("2h")

        self.assertEqual(target_minutes, 120)
        self.assertEqual(fetch_granularity, 3600)

    def test_get_coinbase_fetch_granularity_has_6h(self):
        target_minutes, fetch_granularity = get_coinbase_fetch_granularity("6h")

        self.assertEqual(target_minutes, 360)
        self.assertEqual(fetch_granularity, 21600)

    def test_coinbase_candles_resamples_to_requested_2h_resolution(self):
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
                    "notional": Decimal("10"),
                }
                for hour in range(2)
            ]
        ).set_index("timestamp")

        with patch(
            "quant_tick.exchanges.coinbase.candles.fetch_coinbase_candles",
            return_value=source_candles,
        ) as fetch:
            result = coinbase_candles(
                "BTC-USD",
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        fetch.assert_called_once_with(
            "BTC-USD",
            timestamp_from,
            timestamp_to,
            granularity=3600,
            log_format=None,
        )
        self.assertEqual(list(result.index), [timestamp_from])
        candle = result.iloc[0]
        self.assertEqual(candle.open, Decimal("1"))
        self.assertEqual(candle.high, Decimal("12"))
        self.assertEqual(candle.low, Decimal("0"))
        self.assertEqual(candle.close, Decimal("3"))
        self.assertEqual(candle.notional, Decimal("20"))

    def test_fetch_coinbase_candles_chunks_large_range(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 6, tzinfo=UTC)
        first_chunk = [
            [
                int((timestamp_from + pd.Timedelta("4h59min")).timestamp()),
                "0",
                "2",
                "1",
                "1.5",
                "10",
            ],
            [
                int(timestamp_from.timestamp()),
                "0",
                "1",
                "0.5",
                "0.75",
                "5",
            ],
        ]
        second_chunk = [
            [
                int((timestamp_from + pd.Timedelta("5h59min")).timestamp()),
                "0",
                "4",
                "3",
                "3.5",
                "12",
            ],
            [
                int((timestamp_from + pd.Timedelta("5h")).timestamp()),
                "0",
                "3",
                "2.5",
                "2.75",
                "7",
            ],
        ]

        with patch(
            "quant_tick.exchanges.coinbase.candles.iter_api",
            side_effect=[
                (first_chunk, True, None),
                (second_chunk, True, None),
            ],
        ) as mocked:
            result = fetch_coinbase_candles(
                "BTC-USD",
                timestamp_from,
                timestamp_to,
                granularity=60,
            )

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args_list[0].kwargs["timestamp_from"], timestamp_from)
        self.assertEqual(
            mocked.call_args_list[0].kwargs["pagination_id"],
            "2026-04-01T04:59:00",
        )
        self.assertEqual(
            mocked.call_args_list[1].kwargs["timestamp_from"],
            timestamp_from + pd.Timedelta("5h"),
        )
        self.assertEqual(
            mocked.call_args_list[1].kwargs["pagination_id"],
            "2026-04-01T05:59:00",
        )
        self.assertEqual(
            list(result.index),
            [
                timestamp_from,
                timestamp_from + pd.Timedelta("4h59min"),
                timestamp_from + pd.Timedelta("5h"),
                timestamp_from + pd.Timedelta("5h59min"),
            ],
        )

    def test_candles_api_passes_resolution_to_coinbase(self):
        symbol = SimpleNamespace(exchange=Exchange.COINBASE, api_symbol="BTC-USD")
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 2, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.coinbase_candles",
            return_value=expected,
        ) as mocked:
            result = candles_api(
                symbol,
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        mocked.assert_called_once_with(
            "BTC-USD",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="2h",
        )
        self.assertTrue(result.equals(expected))
