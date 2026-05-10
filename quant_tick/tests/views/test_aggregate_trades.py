from datetime import UTC, datetime
from unittest.mock import patch

import httpx
from django.test import TestCase
from django.urls import reverse

from quant_tick.constants import Exchange, Frequency, TaskType
from quant_tick.lib.download import ArchiveDownloadError
from quant_tick.models import Symbol, TaskState, TradeData
from quant_tick.pubsub import PubSubIngestionResult


@patch("quant_tick.views.aggregate_trades.api")
class AggregateTradeViewTest(TestCase):
    def setUp(self):
        super().setUp()
        self.symbols = {}
        for api_symbol in ("test-1", "test-2"):
            self.symbols[api_symbol] = Symbol.objects.create(
                exchange=Exchange.COINBASE,
                api_symbol=api_symbol,
            )
        Symbol.objects.create(
            exchange=Exchange.BINANCE,
            api_symbol="other-exchange",
        )

    def get_url(self, exchange: str = Exchange.COINBASE) -> str:
        return reverse("aggregate_trades", kwargs={"exchange": exchange})

    def create_trade_data(
        self,
        api_symbol: str = "test-1",
        timestamp: datetime = datetime(2026, 5, 1, tzinfo=UTC),
        frequency: Frequency = Frequency.MINUTE,
        ok: bool | None = True,
    ) -> TradeData:
        return TradeData.objects.create(
            symbol=self.symbols[api_symbol],
            timestamp=timestamp,
            frequency=frequency,
            ok=ok,
        )

    def test_get_with_exchange_processes_all_symbols(self, mock_api):
        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_api.call_count, 4)
        self.assertEqual(
            [(call.args[0].api_symbol, call.args[3]) for call in mock_api.call_args_list],
            [
                ("test-1", True),
                ("test-1", False),
                ("test-2", True),
                ("test-2", False),
            ],
        )
        task_states = TaskState.objects.filter(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
        ).order_by("api_symbol")
        self.assertEqual(
            [task_state.api_symbol for task_state in task_states],
            ["test-1", "test-2"],
        )
        for task_state in task_states:
            self.assertEqual(task_state.recent_error_count, 0)
            self.assertIsNone(task_state.next_fetch_at)
            self.assertIsNone(task_state.locked_until)
        self.assertFalse(
            TaskState.objects.filter(
                task_type=TaskType.AGGREGATE_TRADES,
                exchange=Exchange.COINBASE,
                api_symbol="",
            ).exists()
        )

    def test_get_floors_time_ago_start_to_day_boundary(self, mock_api):
        now = datetime(2026, 5, 2, 0, 10, 42, tzinfo=UTC)
        with patch("quant_tick.views.aggregate_trades.get_current_time", return_value=now):
            response = self.client.get(
                self.get_url(),
                {"time_ago": "7d", "api_symbol": "test-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_api.call_count, 2)
        _symbol, retry_from, retry_to, retry = mock_api.call_args_list[0].args
        self.assertEqual(retry_from, datetime(2026, 5, 1, 23, 10, tzinfo=UTC))
        self.assertEqual(retry_to, datetime(2026, 5, 2, 0, 10, tzinfo=UTC))
        self.assertTrue(retry)
        _symbol, timestamp_from, timestamp_to, retry = mock_api.call_args_list[1].args
        self.assertEqual(timestamp_from, datetime(2026, 4, 25, tzinfo=UTC))
        self.assertEqual(timestamp_to, datetime(2026, 5, 2, 0, 10, tzinfo=UTC))
        self.assertFalse(retry)
        self.assertEqual(
            response.json(),
            {
                "ok": True,
                "exchange": Exchange.COINBASE,
                "api_symbol": "test-1",
            },
        )
        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            api_symbol="test-1",
        )
        self.assertIsNone(task_state.locked_until)

    def test_get_ignores_retry_query_param(self, mock_api):
        now = datetime(2026, 5, 2, 0, 10, 42, tzinfo=UTC)
        with patch("quant_tick.views.aggregate_trades.get_current_time", return_value=now):
            response = self.client.get(
                self.get_url(),
                {"time_ago": "7d", "api_symbol": "test-1", "retry": "true"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_api.call_count, 2)
        _symbol, retry_from, retry_to, retry = mock_api.call_args_list[0].args
        self.assertEqual(retry_from, datetime(2026, 5, 1, 23, 10, tzinfo=UTC))
        self.assertEqual(retry_to, datetime(2026, 5, 2, 0, 10, tzinfo=UTC))
        self.assertTrue(retry)
        _symbol, timestamp_from, timestamp_to, retry = mock_api.call_args_list[1].args
        self.assertEqual(timestamp_from, datetime(2026, 4, 25, tzinfo=UTC))
        self.assertEqual(timestamp_to, datetime(2026, 5, 2, 0, 10, tzinfo=UTC))
        self.assertFalse(retry)
        self.assertEqual(
            response.json(),
            {
                "ok": True,
                "exchange": Exchange.COINBASE,
                "api_symbol": "test-1",
            },
        )

    def test_get_returns_candle_retry_request_for_recent_bad_trade_data(self, mock_api):
        now = datetime(2026, 5, 2, 0, 10, 42, tzinfo=UTC)
        self.create_trade_data(
            timestamp=datetime(2026, 5, 1, 23, 42, tzinfo=UTC),
            ok=False,
        )

        with patch("quant_tick.views.aggregate_trades.get_current_time", return_value=now):
            response = self.client.get(
                self.get_url(),
                {"time_ago": "7d", "api_symbol": "test-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "ok": True,
                "exchange": Exchange.COINBASE,
                "api_symbol": "test-1",
                "timestamp_from": "2026-05-01T23:30:00Z",
            },
        )

    @patch("quant_tick.views.aggregate_trades.ingest_trades_from_pubsub")
    def test_get_skips_pubsub_when_symbol_has_no_trade_data(
        self,
        mock_ingest,
        mock_api,
    ):
        self.symbols["test-1"].significant_trade_filter = 1000
        self.symbols["test-1"].save()

        with patch("quant_tick.pubsub.get_default_project_id", return_value="test"):
            response = self.client.get(self.get_url(), {"api_symbol": "test-1"})

        self.assertEqual(response.status_code, 200)
        mock_ingest.assert_not_called()
        self.assertEqual(mock_api.call_count, 2)

    @patch("quant_tick.views.aggregate_trades.ingest_trades_from_pubsub")
    def test_get_ingests_pubsub_before_rest(self, mock_ingest, mock_api):
        self.symbols["test-1"].significant_trade_filter = 1000
        self.symbols["test-1"].save()
        self.create_trade_data()
        mock_ingest.return_value = PubSubIngestionResult(pulled=1, processed=1, ok=1)

        with patch("quant_tick.pubsub.get_default_project_id", return_value="test"):
            response = self.client.get(self.get_url(), {"api_symbol": "test-1"})

        self.assertEqual(response.status_code, 200)
        mock_ingest.assert_called_once()
        self.assertEqual(
            mock_ingest.call_args.kwargs["configs"],
            [
                (
                    "significant-trades",
                    "projects/test/subscriptions/"
                    "coinbase-test1-significant-trades-1000",
                )
            ],
        )
        self.assertEqual(mock_ingest.call_args.kwargs["symbol"], self.symbols["test-1"])
        self.assertEqual(mock_api.call_count, 2)
        self.assertTrue(mock_api.call_args_list[0].args[3])
        self.assertFalse(mock_api.call_args_list[1].args[3])

    @patch("quant_tick.views.aggregate_trades.ingest_trades_from_pubsub")
    def test_get_skips_pubsub_when_symbol_has_no_trade_streams(
        self,
        mock_ingest,
        mock_api,
    ):
        self.create_trade_data()
        mock_ingest.return_value = PubSubIngestionResult(pulled=1, processed=1, ok=1)

        with patch("quant_tick.pubsub.get_default_project_id", return_value="test"):
            response = self.client.get(self.get_url(), {"api_symbol": "test-1"})

        self.assertEqual(response.status_code, 200)
        mock_ingest.assert_not_called()

    @patch("quant_tick.views.aggregate_trades.ingest_trades_from_pubsub")
    def test_get_uses_all_symbol_pubsub_subscription_contracts(
        self,
        mock_ingest,
        mock_api,
    ):
        self.symbols["test-1"].save_raw = True
        self.symbols["test-1"].save_aggregated = True
        self.symbols["test-1"].significant_trade_filter = 1000
        self.symbols["test-1"].save()
        self.create_trade_data()
        mock_ingest.return_value = PubSubIngestionResult(pulled=1, processed=1, ok=1)

        with patch("quant_tick.pubsub.get_default_project_id", return_value="test"):
            response = self.client.get(self.get_url(), {"api_symbol": "test-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            mock_ingest.call_args.kwargs["configs"],
            [
                ("raw-trades", "projects/test/subscriptions/coinbase-test1-raw-trades"),
                (
                    "aggregated-trades",
                    "projects/test/subscriptions/coinbase-test1-aggregated-trades",
                ),
                (
                    "significant-trades",
                    "projects/test/subscriptions/"
                    "coinbase-test1-significant-trades-1000",
                ),
            ],
        )

    @patch("quant_tick.views.aggregate_trades.ingest_trades_from_pubsub")
    def test_get_scopes_pubsub_to_forward_tail(self, mock_ingest, mock_api):
        now = datetime(2026, 5, 2, 0, 10, 42, tzinfo=UTC)
        self.symbols["test-1"].significant_trade_filter = 1000
        self.symbols["test-1"].save()
        self.create_trade_data(
            timestamp=datetime(2026, 5, 2, 0, 8, tzinfo=UTC),
            frequency=Frequency.MINUTE,
        )
        mock_ingest.return_value = PubSubIngestionResult(pulled=1, processed=1, ok=1)

        with patch("quant_tick.views.aggregate_trades.get_current_time", return_value=now):
            with patch("quant_tick.pubsub.get_default_project_id", return_value="test"):
                response = self.client.get(
                    self.get_url(),
                    {"time_ago": "7d", "api_symbol": "test-1"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            mock_ingest.call_args.kwargs["timestamp_from"],
            datetime(2026, 5, 2, 0, 9, tzinfo=UTC),
        )
        self.assertEqual(
            mock_ingest.call_args.kwargs["timestamp_to"],
            datetime(2026, 5, 2, 0, 10, tzinfo=UTC),
        )

    def test_get_skips_when_task_is_backed_off(self, mock_api):
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            api_symbol="test-1",
            next_fetch_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

        response = self.client.get(self.get_url(), {"api_symbol": "test-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "backoff")
        mock_api.assert_not_called()

    def test_get_skips_when_task_is_locked(self, mock_api):
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            api_symbol="test-1",
            locked_until=datetime(2099, 1, 1, tzinfo=UTC),
        )

        response = self.client.get(self.get_url(), {"api_symbol": "test-1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "locked")
        mock_api.assert_not_called()

    def test_get_with_exchange_skips_locked_symbol(self, mock_api):
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            api_symbol="test-1",
            locked_until=datetime(2099, 1, 1, tzinfo=UTC),
        )

        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_api.call_count, 2)
        self.assertEqual(mock_api.call_args.args[0].api_symbol, "test-2")

    def test_get_marks_error_when_collection_fails(self, mock_api):
        mock_api.side_effect = [None, RuntimeError("boom")]

        with self.assertLogs("django.request", level="ERROR"):
            with self.assertRaises(RuntimeError):
                self.client.get(self.get_url())

        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            api_symbol="test-1",
        )
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNone(task_state.locked_until)

    def test_get_marks_transport_error_without_backoff(self, mock_api):
        mock_api.side_effect = httpx.RemoteProtocolError("server disconnected")

        with self.assertLogs("django.request", level="ERROR"):
            with self.assertRaises(httpx.RemoteProtocolError):
                self.client.get(self.get_url())

        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            api_symbol="test-1",
        )
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNone(task_state.next_fetch_at)
        self.assertIsNone(task_state.locked_until)

    def test_get_does_not_mark_error_for_archive_download_failure(self, mock_api):
        mock_api.side_effect = ArchiveDownloadError("archive boom")

        with self.assertLogs("django.request", level="ERROR"):
            with self.assertRaises(ArchiveDownloadError):
                self.client.get(self.get_url())

        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            api_symbol="test-1",
        )
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
        self.assertIsNone(task_state.locked_until)
