from datetime import UTC, datetime
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from quant_tick.constants import Exchange
from quant_tick.models import Candle, Symbol


class CompactViewTest(TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )
        self.candle = Candle.objects.create(symbol=self.symbol)

    def get_url(self) -> str:
        return reverse("compact")

    def test_get_compacts_all_symbols_and_candles(self):
        with patch(
            "quant_tick.views.compact.convert_trade_data_to_daily"
        ) as mock_compact_trades:
            with patch(
                "quant_tick.views.compact.convert_candle_cache_to_daily"
            ) as mock_compact_candles:
                response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        mock_compact_trades.assert_called_once()
        self.assertEqual(mock_compact_trades.call_args.args[0], self.symbol)
        mock_compact_candles.assert_called_once_with(self.candle)

    def test_get_logs_compaction_errors_and_continues(self):
        with patch(
            "quant_tick.views.compact.convert_trade_data_to_daily",
            side_effect=RuntimeError("boom"),
        ) as mock_compact_trades:
            with patch(
                "quant_tick.views.compact.convert_candle_cache_to_daily"
            ) as mock_compact_candles:
                with self.assertLogs("quant_tick.views.compact", level="ERROR"):
                    response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        mock_compact_trades.assert_called_once()
        mock_compact_candles.assert_called_once_with(self.candle)

    def test_get_compacts_back_max_seven_days(self):
        timestamp_to = datetime(2013, 1, 20, tzinfo=UTC)
        with patch(
            "quant_tick.views.compact.get_request_params",
            return_value=(datetime(2013, 1, 1, tzinfo=UTC), timestamp_to, False),
        ):
            with patch(
                "quant_tick.views.compact.convert_trade_data_to_daily"
            ) as mock_compact_trades:
                with patch("quant_tick.views.compact.convert_candle_cache_to_daily"):
                    response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            mock_compact_trades.call_args.args[1],
            datetime(2013, 1, 13, tzinfo=UTC),
        )
        self.assertEqual(mock_compact_trades.call_args.args[2], timestamp_to)
