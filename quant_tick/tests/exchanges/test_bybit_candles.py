from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase

from quant_tick.constants import SymbolType
from quant_tick.exchanges.bybit.candles import (
    bybit_candles,
    fetch_bybit_candles,
    get_bybit_fetch_resolution,
)


def candle(timestamp, volume="1", turnover="100"):
    return [
        str(int(timestamp.timestamp() * 1000)),
        "100",
        "110",
        "90",
        "105",
        volume,
        turnover,
    ]


class BybitCandleTest(SimpleTestCase):
    def test_fetch_paginates_newest_to_oldest(self):
        timestamp_from = datetime(2026, 7, 23, tzinfo=UTC)
        timestamps = [timestamp_from + timedelta(minutes=value) for value in range(5)]
        pages = [
            {"list": [candle(timestamps[4]), candle(timestamps[3])]},
            {"list": [candle(timestamps[2]), candle(timestamps[1])]},
            {"list": [candle(timestamps[0])]},
        ]

        with patch(
            "quant_tick.exchanges.bybit.candles.get_bybit_candle_response",
            side_effect=pages,
        ) as response:
            result = fetch_bybit_candles(
                "BTCUSDT",
                timestamp_from,
                timestamp_from + timedelta(minutes=5),
                source_minutes=1,
                interval="1",
                category="linear",
            )

        self.assertEqual(result.index.tolist(), timestamps)
        self.assertEqual(response.call_count, 3)
        end_values = [call.args[2] for call in response.call_args_list]
        self.assertEqual(
            end_values,
            [
                int(timestamps[4].timestamp() * 1000),
                int(timestamps[3].timestamp() * 1000) - 1,
                int(timestamps[1].timestamp() * 1000) - 1,
            ],
        )

    def test_linear_candles_map_base_and_quote_units(self):
        timestamp_from = datetime(2026, 7, 23, tzinfo=UTC)
        with patch(
            "quant_tick.exchanges.bybit.candles.get_bybit_candle_response",
            return_value={"list": [candle(timestamp_from, "2", "210")]},
        ):
            result = fetch_bybit_candles(
                "BTCUSDT",
                timestamp_from,
                timestamp_from + timedelta(minutes=1),
                source_minutes=1,
                interval="1",
                category="linear",
            )

        self.assertEqual(result.iloc[0]["notional"], Decimal("2"))
        self.assertEqual(result.iloc[0]["volume"], Decimal("210"))

    def test_inverse_candles_map_contract_and_base_units(self):
        timestamp_from = datetime(2026, 7, 23, tzinfo=UTC)
        with patch(
            "quant_tick.exchanges.bybit.candles.get_bybit_candle_response",
            return_value={"list": [candle(timestamp_from, "210", "2")]},
        ):
            result = fetch_bybit_candles(
                "BTCUSD",
                timestamp_from,
                timestamp_from + timedelta(minutes=1),
                source_minutes=1,
                interval="1",
                category="inverse",
            )

        self.assertEqual(result.iloc[0]["notional"], Decimal("2"))
        self.assertEqual(result.iloc[0]["volume"], Decimal("210"))

    def test_8h_uses_4h_source_resolution(self):
        self.assertEqual(get_bybit_fetch_resolution("8h"), (480, 240, "240"))

    def test_bybit_candles_passes_perpetual_category(self):
        timestamp_from = datetime(2026, 7, 23, tzinfo=UTC)
        with patch(
            "quant_tick.exchanges.bybit.candles.fetch_bybit_candles"
        ) as fetch:
            bybit_candles(
                "BTCUSDT",
                timestamp_from,
                timestamp_from + timedelta(minutes=1),
                symbol_type=SymbolType.PERPETUAL,
            )

        self.assertEqual(fetch.call_args.kwargs["category"], "linear")
