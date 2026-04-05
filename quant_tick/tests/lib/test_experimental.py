from decimal import Decimal

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.lib.experimental import aggregate_distribution, merge_distributions


class ExperimentalDistributionTest(SimpleTestCase):
    """Experimental distribution helper tests."""

    def get_data_frame(
        self,
        prices: list[int | float],
        volumes: list[int | float],
        tick_rules: list[int],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "price": [Decimal(str(price)) for price in prices],
                "volume": [Decimal(str(volume)) for volume in volumes],
                "notional": [
                    Decimal(str(price * volume))
                    for price, volume in zip(prices, volumes, strict=True)
                ],
                "tickRule": tick_rules,
            }
        )

    def test_aggregate_distribution_single_price_level(self):
        df = self.get_data_frame(
            prices=[100, 100, 100],
            volumes=[1, 2, 3],
            tick_rules=[1, -1, 1],
        )

        result = aggregate_distribution(df)

        self.assertEqual(len(result), 1)
        level = list(result.values())[0]
        self.assertEqual(level["ticks"], 3)
        self.assertEqual(level["buyTicks"], 2)
        self.assertEqual(level["volume"], Decimal("6"))
        self.assertEqual(level["buyVolume"], Decimal("4"))

    def test_aggregate_distribution_multiple_price_levels(self):
        df = self.get_data_frame(prices=[100, 200], volumes=[1, 2], tick_rules=[1, 1])

        result = aggregate_distribution(df)

        self.assertGreaterEqual(len(result), 2)
        total_ticks = sum(level["ticks"] for level in result.values())
        self.assertEqual(total_ticks, 2)

    def test_aggregate_distribution_no_levels_when_empty(self):
        df = self.get_data_frame(prices=[100], volumes=[0], tick_rules=[1])

        self.assertEqual(aggregate_distribution(df), {})

    def test_merge_distributions(self):
        previous = {
            "0": {
                "ticks": 2,
                "buyTicks": 1,
                "volume": Decimal("10"),
                "buyVolume": Decimal("5"),
                "notional": Decimal("1000"),
                "buyNotional": Decimal("500"),
            },
        }
        current = {
            "0": {
                "ticks": 3,
                "buyTicks": 2,
                "volume": Decimal("15"),
                "buyVolume": Decimal("10"),
                "notional": Decimal("1500"),
                "buyNotional": Decimal("1000"),
            },
            "1": {
                "ticks": 1,
                "buyTicks": 1,
                "volume": Decimal("2"),
                "buyVolume": Decimal("2"),
                "notional": Decimal("200"),
                "buyNotional": Decimal("200"),
            },
        }

        result = merge_distributions(previous, current)

        self.assertEqual(result["0"]["ticks"], 5)
        self.assertEqual(result["0"]["buyTicks"], 3)
        self.assertEqual(result["0"]["volume"], Decimal("25"))
        self.assertEqual(result["0"]["buyVolume"], Decimal("15"))
        self.assertEqual(result["1"]["ticks"], 1)
