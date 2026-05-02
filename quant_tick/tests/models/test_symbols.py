from django.test import SimpleTestCase

from quant_tick.constants import Exchange
from quant_tick.models import Symbol


class SymbolTest(SimpleTestCase):
    def test_symbol_normalizes_bitfinex_spot_symbol(self):
        symbol = Symbol(exchange=Exchange.BITFINEX, api_symbol="tBTCUSD")

        self.assertEqual(symbol.symbol, "BTCUSD")

    def test_symbol_normalizes_bitfinex_perpetual_symbol(self):
        symbol = Symbol(exchange=Exchange.BITFINEX, api_symbol="tBTCF0:USTF0")

        self.assertEqual(symbol.symbol, "BTCUSD")

    def test_symbol_normalizes_bitfinex_perpetual_symbol_without_prefix(self):
        symbol = Symbol(exchange=Exchange.BITFINEX, api_symbol="BTCF0:USTF0")

        self.assertEqual(symbol.symbol, "BTCUSD")
