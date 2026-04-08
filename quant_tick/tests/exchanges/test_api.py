from datetime import UTC, date, datetime
from unittest.mock import Mock, patch

from django.test import TestCase

from quant_tick.exchanges.api import trades_api

from ..base import BaseSymbolTest


class TradesApiTest(BaseSymbolTest, TestCase):
    def test_trades_api_clamps_timestamp_from_to_symbol_date_from(self):
        symbol = self.get_symbol()
        symbol.date_from = date(2013, 1, 20)
        symbol.save()
        on_data_frame = Mock()

        with patch("quant_tick.exchanges.api.coinbase_trades") as mock_trades:
            trades_api(
                symbol,
                datetime(2013, 1, 1, tzinfo=UTC),
                datetime(2013, 1, 21, tzinfo=UTC),
                on_data_frame,
            )

        mock_trades.assert_called_once()
        self.assertEqual(mock_trades.call_args.args[0], symbol)
        self.assertEqual(
            mock_trades.call_args.kwargs["timestamp_from"],
            datetime(2013, 1, 20, tzinfo=UTC),
        )
        self.assertEqual(
            mock_trades.call_args.kwargs["timestamp_to"],
            datetime(2013, 1, 21, tzinfo=UTC),
        )

    def test_trades_api_skips_before_symbol_date_from(self):
        symbol = self.get_symbol()
        symbol.date_from = date(2013, 1, 20)
        symbol.save()

        with patch("quant_tick.exchanges.api.coinbase_trades") as mock_trades:
            trades_api(
                symbol,
                datetime(2013, 1, 1, tzinfo=UTC),
                datetime(2013, 1, 19, tzinfo=UTC),
                Mock(),
            )

        mock_trades.assert_not_called()
