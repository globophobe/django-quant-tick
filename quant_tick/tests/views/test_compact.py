from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from quant_tick.constants import Exchange, TaskType
from quant_tick.models import Candle, Symbol, TaskState
from quant_tick.views.compact import COMPACT_TASK_API_SYMBOL, COMPACT_TASK_EXCHANGE


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

    def get_task_state(self) -> TaskState:
        return TaskState.objects.get(
            task_type=TaskType.COMPACT,
            exchange=COMPACT_TASK_EXCHANGE,
            api_symbol=COMPACT_TASK_API_SYMBOL,
        )

    def create_task_state(self, **kwargs) -> TaskState:
        data = {
            "task_type": TaskType.COMPACT,
            "exchange": COMPACT_TASK_EXCHANGE,
            "api_symbol": COMPACT_TASK_API_SYMBOL,
        }
        data.update(kwargs)
        return TaskState.objects.create(**data)

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
        task_state = self.get_task_state()
        self.assertIsNone(task_state.locked_until)
        self.assertEqual(task_state.recent_error_count, 0)

    def test_get_skips_when_compact_task_is_locked(self):
        self.create_task_state(locked_until=timezone.now() + timedelta(minutes=5))
        with patch(
            "quant_tick.views.compact.convert_trade_data_to_daily"
        ) as mock_compact_trades:
            with patch(
                "quant_tick.views.compact.convert_candle_cache_to_daily"
            ) as mock_compact_candles:
                response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "skipped": "locked"})
        mock_compact_trades.assert_not_called()
        mock_compact_candles.assert_not_called()

    def test_get_skips_when_compact_task_is_backed_off(self):
        self.create_task_state(next_fetch_at=timezone.now() + timedelta(minutes=5))
        with patch(
            "quant_tick.views.compact.convert_trade_data_to_daily"
        ) as mock_compact_trades:
            with patch(
                "quant_tick.views.compact.convert_candle_cache_to_daily"
            ) as mock_compact_candles:
                response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "skipped": "backoff"})
        mock_compact_trades.assert_not_called()
        mock_compact_candles.assert_not_called()

    def test_get_logs_compaction_errors_continues_and_marks_task_error(self):
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
        task_state = self.get_task_state()
        self.assertIsNone(task_state.locked_until)
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNotNone(task_state.recent_error_at)
        self.assertIsNone(task_state.next_fetch_at)

    def test_get_compacts_back_max_seven_days_and_skips_recent_two_hours(self):
        timestamp_to = datetime(2013, 1, 20, 0, 10, tzinfo=UTC)
        with patch(
            "quant_tick.views.compact.get_request_params",
            return_value=(datetime(2013, 1, 1, tzinfo=UTC), timestamp_to),
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
        self.assertEqual(
            mock_compact_trades.call_args.args[2],
            datetime(2013, 1, 19, 22, tzinfo=UTC),
        )
