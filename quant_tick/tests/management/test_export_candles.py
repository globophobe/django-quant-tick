from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Frequency
from quant_tick.lib import aggregate_candle
from quant_tick.management.commands.export_candles import get_output_path
from quant_tick.models import AdaptiveCandle, CandleData, Symbol, TimeBasedCandle


class ExportCandlesFilenameTest(SimpleTestCase):
    def test_get_output_path_for_time_based_candle(self) -> None:
        candle = TimeBasedCandle(
            symbol=Symbol(exchange="binance", api_symbol="BTCUSDT"),
            json_data={
                "window": "1d",
                "source_data": "filtered_data",
            },
        )

        output_path = get_output_path(candle, "20260421")

        self.assertEqual(
            output_path,
            "binance-btcusdt-timebased-1d-filtered-candles-20260421.parquet",
        )
        candle.json_data["volume_brackets"] = [10000, 100000]
        self.assertEqual(
            get_output_path(candle, "20260421"),
            "binance-btcusdt-timebased-1d-filtered-vb10000-100000-candles-20260421.parquet",
        )

    def test_get_output_path_for_adaptive_candle_with_cache_reset(self) -> None:
        candle = AdaptiveCandle(
            symbol=Symbol(exchange="binance", api_symbol="BTCUSDT"),
            json_data={
                "source_data": "filtered_data",
                "sample_type": "notional",
                "target_candles_per_day": 50,
                "moving_average_number_of_days": 50,
                "min_volume_exponent": 1,
                "min_notional_exponent": 1,
                "cache_reset": Frequency.WEEK,
            },
        )

        output_path = get_output_path(candle, "20260421")

        self.assertEqual(
            output_path,
            "binance-btcusdt-adaptive-filtered-notional-50cpd-ma50d-round-weekly-cache-reset-candles-20260421.parquet",
        )
        candle.json_data["volume_brackets"] = [1000, 10000]
        self.assertEqual(
            get_output_path(candle, "20260421"),
            "binance-btcusdt-adaptive-filtered-notional-50cpd-ma50d-round-vb1000-10000-weekly-cache-reset-candles-20260421.parquet",
        )

    def test_get_output_path_for_adaptive_candle_with_calendar_cache_reset(
        self,
    ) -> None:
        candle = AdaptiveCandle(
            symbol=Symbol(exchange="binance", api_symbol="BTCUSDT"),
            json_data={
                "source_data": "filtered_data",
                "sample_type": "notional",
                "target_candles_per_day": 50,
                "moving_average_number_of_days": 50,
                "cache_reset": "quarter",
            },
        )

        output_path = get_output_path(candle, "20260421")

        self.assertEqual(
            output_path,
            "binance-btcusdt-adaptive-filtered-notional-50cpd-ma50d-quarter-cache-reset-candles-20260421.parquet",
        )


class ExportBracketCandlesTest(TestCase):
    def test_stored_brackets_export_across_years_with_empty_buckets(self):
        symbol = Symbol.objects.create(exchange="coinbase", api_symbol="BTC-USD")
        candle = TimeBasedCandle.objects.create(
            symbol=symbol,
            json_data={
                "window": "1d",
                "source_data": "raw_data",
                "volume_brackets": [10000, 100000],
            },
        )
        for year, volume, side in ((2024, 500, 1), (2025, 100000, -1)):
            raw = pd.DataFrame(
                [
                    {
                        "timestamp": datetime(year, 1, 1, tzinfo=UTC),
                        "price": Decimal(1000),
                        "volume": Decimal(volume),
                        "notional": Decimal(volume) / 1000,
                        "tickRule": side,
                    }
                ]
            )
            payload = aggregate_candle(
                raw, volume_brackets=candle.json_data["volume_brackets"]
            )
            CandleData.objects.from_data(candle, payload).save()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "candles.parquet"
            with patch(
                "quant_tick.management.commands.export_candles.get_output_path",
                return_value=str(output),
            ):
                call_command(
                    "export_candles", code_name=candle.code_name, stdout=StringIO()
                )
            rows = pq.read_table(output).to_pylist()
        self.assertEqual(len(rows), 2)
        first, second = [row["volumeBrackets"] for row in rows]
        self.assertEqual(first["buckets"][0]["buyVolume"], 500)
        self.assertEqual(first["buckets"][2]["volume"], 0)
        self.assertEqual(second["buckets"][2]["volume"], 100000)
        self.assertEqual(second["buckets"][2]["buyVolume"], 0)
        self.assertIsNone(second["buckets"][2]["maxVolume"])
        self.assertEqual(second["unclassified"]["volume"], 0)
