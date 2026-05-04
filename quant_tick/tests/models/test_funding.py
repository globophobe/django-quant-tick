from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from django.test import TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.models import FundingData

from ..base import BaseSymbolTest


class FundingDataTest(BaseSymbolTest, TestCase):
    def test_write_replaces_perpetual_funding_rows_in_range(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 2, tzinfo=UTC)
        data = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from,
                    "funding_rate": Decimal("0.0001"),
                    "mark_price": Decimal("95000"),
                },
                {
                    "timestamp": timestamp_from + pd.Timedelta("1h"),
                    "funding_rate": Decimal("0.0002"),
                    "premium": Decimal("0.00003"),
                    "source": "test",
                },
            ]
        )

        FundingData.write(symbol, timestamp_from, timestamp_to, data)
        FundingData.write(symbol, timestamp_from, timestamp_to, data.iloc[:1])

        rows = list(FundingData.objects.filter(symbol=symbol))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, timestamp_from)
        self.assertEqual(rows[0].funding_rate, Decimal("0.0001"))
        self.assertEqual(rows[0].json_data["mark_price"], Decimal("95000"))
        self.assertEqual(rows[0].to_row()["mark_price"], Decimal("95000"))

    def test_write_rejects_spot_symbol(self):
        symbol = self.get_symbol(symbol_type=SymbolType.SPOT)
        data = pd.DataFrame(
            [
                {
                    "timestamp": datetime(2026, 4, 25, tzinfo=UTC),
                    "funding_rate": Decimal("0.0001"),
                }
            ]
        )

        with self.assertRaises(ValueError):
            FundingData.write(
                symbol,
                datetime(2026, 4, 25, tzinfo=UTC),
                datetime(2026, 4, 26, tzinfo=UTC),
                data,
            )
