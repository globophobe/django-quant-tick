from decimal import Decimal

import pandas as pd

from quant_tick.models import (
    CandleData,
    Renko2BrickReversalStrategy,
    RenkoBrick,
    RenkoData,
)

from ..test_strategies import BaseStrategyTest


class Renko2BrickReversalStrategyTest(BaseStrategyTest):
    def setUp(self):
        """Set up."""
        super().setUp()
        self.candle = RenkoBrick.objects.create(symbol=self.symbol)
        self.strategy = Renko2BrickReversalStrategy.objects.create(
            candle=self.candle,
            json_data={"cost": "0"},
        )

    def _create_renko_rows(self, directions: list[int], closes: list[int]) -> None:
        for i, (direction, close) in enumerate(zip(directions, closes, strict=True)):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": close},
            )
            RenkoData.objects.create(
                candle_data=candle_data,
                level=i,
                sequence=i,
                direction=direction,
            )

    def test_get_events_with_exit(self):
        """Confirmed entries and exit on next reversal are captured."""
        directions = [1, 1, -1, -1, 1, 1]
        closes = [100, 101, 99, 98, 99, 100]
        self._create_renko_rows(directions, closes)

        timestamp_to = self.timestamp_from + pd.Timedelta(f"{len(directions) + 1}min")
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=timestamp_to,
            include_incomplete=False,
        )

        self.assertEqual(len(events), 1)
        down_evt = events.iloc[0]

        self.assertEqual(down_evt["direction"], -1)
        self.assertEqual(down_evt["entry_price"], Decimal("98"))
        self.assertEqual(down_evt["exit_price"], Decimal("99"))
        expected_gross = -1 * (Decimal("99") / Decimal("98") - 1)
        self.assertAlmostEqual(float(down_evt["gross_return"]), float(expected_gross))
        self.assertEqual(down_evt["run_length_prev"], 2)
        self.assertIsNotNone(down_evt["candle_data_id"])
