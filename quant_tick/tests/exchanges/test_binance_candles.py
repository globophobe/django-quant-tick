from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import candles_api
from quant_tick.exchanges.binance.candles import (
    binance_candles,
    fetch_binance_candles,
    get_binance_candle_pagination_id,
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
        source_candles = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + pd.Timedelta(f"{hour * 2}h"),
                    "open": Decimal(str(hour + 1)),
                    "high": Decimal(str(hour + 3)),
                    "low": Decimal(str(hour)),
                    "close": Decimal(str(hour + 2)),
                    "notional": Decimal("10"),
                }
                for hour in range(2)
            ]
        ).set_index("timestamp")

        with patch(
            "quant_tick.exchanges.binance.candles.fetch_binance_candles",
            return_value=source_candles,
        ) as mocked:
            result = binance_candles(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
                resolution="2h",
            )

        self.assertEqual(list(result.index), [timestamp_from, timestamp_from + pd.Timedelta("2h")])
        self.assertEqual(result.iloc[0].open, Decimal("1"))
        self.assertEqual(result.iloc[1].close, Decimal("3"))
        mocked.assert_called_once_with(
            "BTCUSDT",
            timestamp_from,
            timestamp_to,
            interval="2h",
            symbol_type=SymbolType.SPOT,
            limit=1000,
            log_format=None,
        )

    def test_get_binance_candle_pagination_id_steps_before_oldest_open(self):
        timestamp = datetime(2026, 4, 1, 5, tzinfo=UTC)

        pagination_id = get_binance_candle_pagination_id(timestamp)

        self.assertEqual(pagination_id, int(timestamp.timestamp() * 1000) - 1)

    def test_fetch_binance_candles_backfills_with_iter_api(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 6, tzinfo=UTC)
        results = [
            [
                int((timestamp_from + pd.Timedelta("5h")).timestamp() * 1000),
                "2.5",
                "4",
                "2",
                "3",
                "7",
            ],
            [
                int((timestamp_from + pd.Timedelta("1h")).timestamp() * 1000),
                "1.5",
                "3",
                "1",
                "2",
                "12",
            ],
            [
                int(timestamp_from.timestamp() * 1000),
                "0.5",
                "1",
                "0.25",
                "0.75",
                "5",
            ],
        ]

        with patch(
            "quant_tick.exchanges.binance.candles.iter_api",
            return_value=(results, False, int(timestamp_from.timestamp() * 1000) - 1),
        ) as mocked:
            result = fetch_binance_candles(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
                interval="1h",
                limit=5,
            )

        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.kwargs["timestamp_from"], timestamp_from)
        self.assertEqual(
            mocked.call_args.kwargs["pagination_id"],
            int((timestamp_to - pd.Timedelta("1h")).timestamp() * 1000),
        )
        self.assertEqual(
            list(result.index),
            [
                timestamp_from,
                timestamp_from + pd.Timedelta("1h"),
                timestamp_from + pd.Timedelta("5h"),
            ],
        )

    def test_fetch_binance_candles_uses_futures_api_for_perpetual(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 1, tzinfo=UTC)

        with patch(
            "quant_tick.exchanges.binance.candles.iter_api",
            return_value=([], False, None),
        ) as mocked:
            fetch_binance_candles(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
                interval="1h",
                symbol_type=SymbolType.PERPETUAL,
            )

        url = mocked.call_args.args[0]
        self.assertTrue(url.startswith("https://fapi.binance.com/fapi/v1/klines?"))

    def test_candles_api_passes_resolution_to_binance(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
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
            symbol_type=SymbolType.PERPETUAL,
        )
        self.assertTrue(result.equals(expected))
