from datetime import UTC, date, datetime

from django.test import TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.management.base import BaseTradeDataCommand

from ..base import BaseSymbolTest


class BaseTradeDataCommandTest(BaseSymbolTest, TestCase):
    def get_options(self, date_to: str = "2013-01-21") -> dict:
        return {
            "exchange": None,
            "api_symbol": None,
            "code_name": None,
            "symbol_type": None,
            "date_from": "2013-01-01",
            "time_from": None,
            "date_to": date_to,
            "time_to": None,
        }

    def test_handle_clamps_timestamp_from_to_symbol_date_from(self):
        symbol = self.get_symbol()
        symbol.date_from = date(2013, 1, 20)
        symbol.save()

        jobs = list(BaseTradeDataCommand().handle(**self.get_options()))

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["symbol"], symbol)
        self.assertEqual(jobs[0]["timestamp_from"], datetime(2013, 1, 20, tzinfo=UTC))
        self.assertEqual(jobs[0]["timestamp_to"], datetime(2013, 1, 21, tzinfo=UTC))

    def test_handle_skips_symbol_before_date_from(self):
        symbol = self.get_symbol()
        symbol.date_from = date(2013, 1, 20)
        symbol.save()

        jobs = list(BaseTradeDataCommand().handle(**self.get_options("2013-01-19")))

        self.assertEqual(jobs, [])

    def test_handle_filters_symbols(self):
        match = self.get_symbol(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        self.get_symbol(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT,
        )
        self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            symbol_type=SymbolType.SPOT,
        )
        options = self.get_options()
        options.update(
            {
                "exchange": [Exchange.BINANCE],
                "api_symbol": ["BTCUSDT"],
                "code_name": [match.code_name],
                "symbol_type": [SymbolType.PERPETUAL],
            }
        )

        jobs = list(BaseTradeDataCommand().handle(**options))

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["symbol"], match)
