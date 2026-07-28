from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import exchange_candles_api
from quant_tick.exchanges.deribit.api import get_deribit_result
from quant_tick.exchanges.deribit.candles import (
    deribit_candles,
    get_deribit_fetch_resolution,
)


class DeribitCandleTest(SimpleTestCase):
    def test_get_deribit_fetch_resolution_resamples_8h_from_2h(self):
        self.assertEqual(
            get_deribit_fetch_resolution("8h"),
            (480, 120, "120"),
        )

    def test_deribit_candles_resamples_and_maps_exchange_units(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=8)
        ticks = [
            int((timestamp_from + timedelta(hours=offset)).timestamp() * 1000)
            for offset in (0, 2, 4, 6)
        ]
        response = {
            "status": "ok",
            "ticks": ticks,
            "open": ["100", "101", "102", "103"],
            "high": ["102", "103", "104", "105"],
            "low": ["99", "100", "101", "102"],
            "close": ["101", "102", "103", "104"],
            "volume": ["1", "2", "3", "4"],
            "cost": ["100", "200", "300", "400"],
        }

        with patch(
            "quant_tick.exchanges.deribit.candles.get_deribit_candle_response",
            return_value=response,
        ) as mocked:
            result = deribit_candles(
                "BTC-PERPETUAL",
                timestamp_from,
                timestamp_to,
                resolution="8h",
            )

        mocked.assert_called_once_with(
            "BTC-PERPETUAL",
            int(timestamp_from.timestamp() * 1000),
            int(timestamp_to.timestamp() * 1000) - 1,
            "120",
        )
        self.assertEqual(list(result.index), [pd.Timestamp(timestamp_from)])
        row = result.iloc[0]
        self.assertEqual(row.open, Decimal("100"))
        self.assertEqual(row.high, Decimal("105"))
        self.assertEqual(row.low, Decimal("99"))
        self.assertEqual(row.close, Decimal("104"))
        self.assertEqual(row.notional, Decimal("10"))
        self.assertEqual(row.volume, Decimal("1000"))

    def test_deribit_candles_uses_non_overlapping_bounded_windows(self):
        timestamp_from = datetime(2026, 1, 1, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=1001)

        with patch(
            "quant_tick.exchanges.deribit.candles.get_deribit_candle_response",
            side_effect=[{"status": "no_data"}, {"status": "no_data"}],
        ) as mocked:
            result = deribit_candles(
                "BTC-PERPETUAL",
                timestamp_from,
                timestamp_to,
                resolution="1m",
            )

        start_ms = int(timestamp_from.timestamp() * 1000)
        first_end_ms = start_ms + 999 * 60_000
        self.assertEqual(
            mocked.call_args_list,
            [
                call("BTC-PERPETUAL", start_ms, first_end_ms, "1"),
                call(
                    "BTC-PERPETUAL",
                    first_end_ms + 1,
                    int(timestamp_to.timestamp() * 1000) - 1,
                    "1",
                ),
            ],
        )
        self.assertTrue(result.empty)

    def test_deribit_candles_rejects_misaligned_response_arrays(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=1)
        response = {
            "status": "ok",
            "ticks": [int(timestamp_from.timestamp() * 1000)],
            "open": ["100"],
            "high": ["101"],
            "low": ["99"],
            "close": [],
            "volume": ["1"],
            "cost": ["100"],
        }

        with (
            patch(
                "quant_tick.exchanges.deribit.candles.get_deribit_candle_response",
                return_value=response,
            ),
            self.assertRaisesRegex(ValueError, "arrays differ in length"),
        ):
            deribit_candles(
                "BTC-PERPETUAL",
                timestamp_from,
                timestamp_to,
                resolution="1m",
            )

    def test_exchange_candles_api_dispatches_deribit(self):
        symbol = SimpleNamespace(
            exchange=Exchange.DERIBIT,
            api_symbol="BTC-PERPETUAL",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="8h",
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.deribit_candles",
            return_value=expected,
        ) as mocked:
            result = exchange_candles_api(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            "BTC-PERPETUAL",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="8h",
        )
        self.assertTrue(result.equals(expected))

    def test_deribit_api_uses_public_json_rpc_result_and_throttle(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {"jsonrpc": "2.0", "result": {"status": "ok"}}

        with (
            patch(
                "quant_tick.exchanges.deribit.api.httpx.get",
                return_value=response,
            ) as mocked_get,
            patch(
                "quant_tick.exchanges.deribit.api.time.time",
                side_effect=[0, 0],
            ),
            patch("quant_tick.exchanges.deribit.api.time.sleep") as mocked_sleep,
        ):
            result = get_deribit_result("test", {"instrument_name": "BTC-PERPETUAL"})

        self.assertEqual(result, {"status": "ok"})
        mocked_get.assert_called_once_with(
            "https://www.deribit.com/api/v2/public/test",
            params={"instrument_name": "BTC-PERPETUAL"},
            timeout=30,
        )
        mocked_sleep.assert_called_once_with(0.05)
