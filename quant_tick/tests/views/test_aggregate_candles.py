from datetime import UTC, datetime
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from quant_tick.constants import Exchange, TaskType
from quant_tick.models import Candle, Symbol, TaskState


class AggregateCandleViewTest(TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )

    def get_url(self) -> str:
        return reverse("aggregate_candles")

    def test_get_processes_all_candles(self):
        order = []
        Candle.objects.create(symbol=self.symbol)
        Candle.objects.create(symbol=self.symbol)

        def on_candles(candle, *_args):
            order.append(candle.code_name)

        with patch(
            "quant_tick.models.candles.Candle.candles",
            autospec=True,
            side_effect=on_candles,
        ) as mock_candles:
            response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_candles.call_count, 2)
        self.assertEqual(len(order), 2)
        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange="",
        )
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
        self.assertIsNone(task_state.locked_until)

    def test_get_skips_when_task_is_backed_off(self):
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange="",
            next_fetch_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

        with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
            response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "backoff")
        mock_candles.assert_not_called()

    def test_get_skips_when_task_is_locked(self):
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange="",
            locked_until=datetime(2099, 1, 1, tzinfo=UTC),
        )

        with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
            response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "locked")
        mock_candles.assert_not_called()

    def test_get_marks_error_when_aggregation_fails(self):
        Candle.objects.create(symbol=self.symbol)
        Candle.objects.create(symbol=self.symbol)

        with patch(
            "quant_tick.models.candles.Candle.candles",
            autospec=True,
            side_effect=[None, RuntimeError("boom")],
        ):
            with self.assertLogs("django.request", level="ERROR"):
                with self.assertRaises(RuntimeError):
                    self.client.get(self.get_url())

        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange="",
        )
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNone(task_state.locked_until)
