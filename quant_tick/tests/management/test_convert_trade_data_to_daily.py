from datetime import UTC, datetime
from unittest.mock import patch

from django.test import TestCase

from quant_tick.management.commands.convert_trade_data_to_daily import Command

from ..base import BaseSymbolTest


class ConvertTradeDataToDailyCommandTest(BaseSymbolTest, TestCase):
    def get_options(self) -> dict:
        return {
            "exchange": None,
            "api_symbol": None,
            "code_name": None,
            "symbol_type": None,
            "date_from": "2026-05-01",
            "time_from": None,
            "date_to": "2026-05-02",
            "time_to": "00:10",
        }

    @patch("quant_tick.management.commands.convert_trade_data_to_daily.get_current_time")
    @patch(
        "quant_tick.management.commands.convert_trade_data_to_daily."
        "convert_trade_data_to_daily"
    )
    def test_handle_clamps_timestamp_to_to_compact_max_time(
        self,
        mock_convert_trade_data_to_daily,
        mock_get_current_time,
    ):
        symbol = self.get_symbol()
        mock_get_current_time.return_value = datetime(2026, 5, 2, 0, 10, tzinfo=UTC)

        Command().handle(**self.get_options())

        mock_convert_trade_data_to_daily.assert_called_once()
        self.assertEqual(mock_convert_trade_data_to_daily.call_args.kwargs["symbol"], symbol)
        self.assertEqual(
            mock_convert_trade_data_to_daily.call_args.kwargs["timestamp_from"],
            datetime(2026, 5, 1, tzinfo=UTC),
        )
        self.assertEqual(
            mock_convert_trade_data_to_daily.call_args.kwargs["timestamp_to"],
            datetime(2026, 5, 1, 22, tzinfo=UTC),
        )
