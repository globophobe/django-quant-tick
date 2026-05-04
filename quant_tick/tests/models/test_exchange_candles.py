from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from django.test import TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.models import ExchangeCandleData

from ..base import BaseSymbolTest


class ExchangeCandleDataTest(BaseSymbolTest, TestCase):
    def test_write_replaces_exchange_candles_for_frequency(self):
        symbol = self.get_symbol(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 2, tzinfo=UTC)
        data = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from,
                    "open": Decimal("1"),
                    "high": Decimal("2"),
                    "low": Decimal("0.5"),
                    "close": Decimal("1.5"),
                    "notional": Decimal("10"),
                    "trades": 5,
                },
                {
                    "timestamp": timestamp_from + pd.Timedelta("1h"),
                    "open": Decimal("2"),
                    "high": Decimal("3"),
                    "low": Decimal("1.5"),
                    "close": Decimal("2.5"),
                    "notional": Decimal("11"),
                },
            ]
        )

        ExchangeCandleData.write(symbol, 60, timestamp_from, timestamp_to, data)
        ExchangeCandleData.write(symbol, 60, timestamp_from, timestamp_to, data.iloc[:1])

        rows = list(ExchangeCandleData.objects.filter(symbol=symbol))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, timestamp_from)
        self.assertEqual(rows[0].notional, Decimal("10"))
        self.assertEqual(rows[0].json_data["trades"], 5)
