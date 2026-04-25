from django.test import SimpleTestCase

from quant_tick.management.commands.export_candles import get_output_path
from quant_tick.models import Symbol, TimeBasedCandle


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
