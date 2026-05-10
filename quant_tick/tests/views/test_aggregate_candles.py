import json
from datetime import UTC, datetime
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from quant_tick.constants import Exchange, TaskType
from quant_tick.models import Candle, Symbol, TaskState
from quant_tick.views.aggregate_candles import DEFAULT_TIMESTAMP_FROM


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
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
        self.assertIsNone(task_state.locked_until)
        self.assertFalse(
            TaskState.objects.filter(
                task_type=TaskType.AGGREGATE_CANDLES,
                api_symbol="",
            ).exists()
        )

    def test_get_with_exchange_processes_matching_candles(self):
        coinbase = Candle.objects.create(symbol=self.symbol)
        binance_symbol = Symbol.objects.create(
            exchange=Exchange.BINANCE,
            api_symbol="other-test",
        )
        Candle.objects.create(symbol=binance_symbol)
        order = []

        def on_candles(candle, *_args):
            order.append(candle.code_name)

        with patch(
            "quant_tick.models.candles.Candle.candles",
            autospec=True,
            side_effect=on_candles,
        ) as mock_candles:
            response = self.client.get(self.get_url(), {"exchange": Exchange.COINBASE})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_candles.call_count, 1)
        self.assertEqual(order, [coinbase.code_name])
        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )
        self.assertEqual(task_state.recent_error_count, 0)

    def test_get_with_api_symbol_processes_matching_candles(self):
        coinbase = Candle.objects.create(symbol=self.symbol)
        other_symbol = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="other-test",
        )
        Candle.objects.create(symbol=other_symbol)
        order = []

        def on_candles(candle, *_args):
            order.append(candle.code_name)

        with patch(
            "quant_tick.models.candles.Candle.candles",
            autospec=True,
            side_effect=on_candles,
        ) as mock_candles:
            response = self.client.get(
                self.get_url(),
                {"exchange": Exchange.COINBASE, "api_symbol": "test"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_candles.call_count, 1)
        self.assertEqual(order, [coinbase.code_name])

    def test_get_defaults_to_tail_forward_start(self):
        Candle.objects.create(symbol=self.symbol)
        now = datetime(2026, 5, 9, 12, 30, 45, tzinfo=UTC)

        with patch("quant_tick.views.aggregate_candles.get_current_time", return_value=now):
            with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
                response = self.client.get(
                    self.get_url(),
                    {"exchange": Exchange.COINBASE, "api_symbol": "test"},
                )

        self.assertEqual(response.status_code, 200)
        timestamp_from, timestamp_to, retry = mock_candles.call_args.args
        self.assertEqual(timestamp_from, DEFAULT_TIMESTAMP_FROM)
        self.assertEqual(timestamp_to, datetime(2026, 5, 9, 12, 30, tzinfo=UTC))
        self.assertFalse(retry)

    def test_post_processes_single_candle_request(self):
        Candle.objects.create(symbol=self.symbol)
        now = datetime(2026, 5, 9, 12, 30, 45, tzinfo=UTC)
        body = {
            "exchange": Exchange.COINBASE,
            "api_symbol": "test",
            "timestamp_from": "2026-05-09T11:42:00Z",
        }

        with patch("quant_tick.views.aggregate_candles.get_current_time", return_value=now):
            with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
                response = self.client.post(
                    self.get_url(),
                    data=json.dumps(body),
                    content_type="application/json",
                )

        self.assertEqual(response.status_code, 200)
        timestamp_from, timestamp_to, retry = mock_candles.call_args.args
        self.assertEqual(timestamp_from, datetime(2026, 5, 9, 11, tzinfo=UTC))
        self.assertEqual(timestamp_to, datetime(2026, 5, 9, 12, 30, tzinfo=UTC))
        self.assertTrue(retry)

    def test_post_processes_multiple_candle_requests(self):
        Candle.objects.create(symbol=self.symbol)
        binance_symbol = Symbol.objects.create(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
        )
        Candle.objects.create(symbol=binance_symbol)
        now = datetime(2026, 5, 9, 12, 30, 45, tzinfo=UTC)
        body = {
            "ok": True,
            "candle_requests": [
                {
                    "exchange": Exchange.COINBASE,
                    "api_symbol": "test",
                    "timestamp_from": "2026-05-09T11:42:00Z",
                },
                {
                    "exchange": Exchange.BINANCE,
                    "api_symbol": "BTCUSDT",
                },
            ],
        }

        with patch("quant_tick.views.aggregate_candles.get_current_time", return_value=now):
            with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
                response = self.client.post(
                    self.get_url(),
                    data=json.dumps(body),
                    content_type="application/json",
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["processed"], 2)
        self.assertEqual(mock_candles.call_count, 2)

    def test_post_rejects_invalid_candle_request(self):
        body = {"candle_requests": [{"exchange": "not-an-exchange"}]}

        response = self.client.post(
            self.get_url(),
            data=json.dumps(body),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("exchange", response.json()["error"])

    def test_get_skips_when_task_is_backed_off(self):
        Candle.objects.create(symbol=self.symbol)
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=Exchange.COINBASE,
            api_symbol="test",
            next_fetch_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

        with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
            response = self.client.get(
                self.get_url(),
                {"exchange": Exchange.COINBASE, "api_symbol": "test"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "backoff")
        mock_candles.assert_not_called()

    def test_get_skips_when_task_is_locked(self):
        Candle.objects.create(symbol=self.symbol)
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=Exchange.COINBASE,
            api_symbol="test",
            locked_until=datetime(2099, 1, 1, tzinfo=UTC),
        )

        with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
            response = self.client.get(
                self.get_url(),
                {"exchange": Exchange.COINBASE, "api_symbol": "test"},
            )

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
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNone(task_state.locked_until)
