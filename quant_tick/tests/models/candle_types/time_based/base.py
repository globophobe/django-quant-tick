import pandas as pd

from quant_tick.constants import FileData
from quant_tick.lib import aggregate_candle
from quant_tick.models import TimeBasedCandle
from quant_tick.models.candles import camel_to_snake

from ..base import BaseTradeDataCandleTest


class BaseTimeBasedCandleTest(BaseTradeDataCandleTest):
    window = ""

    def get_candle(self) -> TimeBasedCandle:
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": self.window},
        )

    def assert_combined_candle(self, data_frames: list[pd.DataFrame]) -> None:
        expected = aggregate_candle(pd.concat(data_frames))
        del expected["timestamp"]
        expected = {camel_to_snake(key): value for key, value in expected.items()}
        actual = next(self.candle.get_candle_data())
        del actual["timestamp"]
        self.assertGreater(actual["realized_variance"], 0)
        self.assertGreater(expected["realized_variance"], 0)
        self.assertAlmostEqual(
            float(actual.pop("realized_variance")),
            float(expected.pop("realized_variance")),
            places=12,
        )
        self.assertEqual(actual, expected)
