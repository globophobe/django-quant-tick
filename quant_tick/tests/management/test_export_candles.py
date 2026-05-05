from django.test import SimpleTestCase

from quant_tick.constants import Frequency
from quant_tick.management.commands.export_candles import get_output_path
from quant_tick.models import AdaptiveCandle, Symbol, TimeBasedCandle


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

    def test_get_output_path_for_adaptive_candle_with_calendar_cache_reset(self) -> None:
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
