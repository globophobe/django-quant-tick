from unittest.mock import patch

from django.test import TestCase

from quant_tick.management.commands.exchange_candles import Command

from ..base import BaseSymbolTest


class ExchangeCandlesCommandTest(BaseSymbolTest, TestCase):
    def get_options(self, resolution: str | None = None) -> dict:
        return {
            "exchange": None,
            "api_symbol": None,
            "code_name": None,
            "symbol_type": None,
            "date_from": "2026-04-25",
            "time_from": None,
            "date_to": "2026-04-26",
            "time_to": None,
            "retry": False,
            "resolution": resolution,
        }

    @patch("quant_tick.management.commands.exchange_candles.exchange_candles")
    def test_handle_uses_symbol_exchange_candle_resolution(self, mock_exchange_candles):
        symbol = self.get_symbol(exchange_candle_resolution="4h")

        Command().handle(**self.get_options())

        mock_exchange_candles.assert_called_once()
        self.assertEqual(mock_exchange_candles.call_args.kwargs["symbol"], symbol)
        self.assertEqual(mock_exchange_candles.call_args.kwargs["resolution"], "4h")

    @patch("quant_tick.management.commands.exchange_candles.exchange_candles")
    def test_handle_resolution_overrides_symbol_resolution(self, mock_exchange_candles):
        self.get_symbol(exchange_candle_resolution="4h")

        Command().handle(**self.get_options(resolution="1h"))

        mock_exchange_candles.assert_called_once()
        self.assertEqual(mock_exchange_candles.call_args.kwargs["resolution"], "1h")

    @patch("quant_tick.management.commands.exchange_candles.exchange_candles")
    def test_handle_skips_symbols_without_exchange_candle_resolution(
        self,
        mock_exchange_candles,
    ):
        self.get_symbol()

        Command().handle(**self.get_options())

        mock_exchange_candles.assert_not_called()
