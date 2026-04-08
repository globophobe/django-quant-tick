from datetime import UTC, datetime
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from quant_tick.constants import Exchange, TaskType
from quant_tick.lib.download import ArchiveDownloadError
from quant_tick.models import Symbol, TaskState


@patch("quant_tick.views.aggregate_trades.api")
class AggregateTradeViewTest(TestCase):
    def setUp(self):
        super().setUp()
        for api_symbol in ("test-1", "test-2"):
            Symbol.objects.create(
                exchange=Exchange.COINBASE,
                api_symbol=api_symbol,
            )
        Symbol.objects.create(
            exchange=Exchange.BINANCE,
            api_symbol="other-exchange",
        )

    def get_url(self, exchange: str = Exchange.COINBASE) -> str:
        return reverse("aggregate_trades", kwargs={"exchange": exchange})

    def test_get_with_exchange_processes_all_symbols(self, mock_api):
        order = []

        def on_api(symbol, *_args):
            order.append(symbol.api_symbol)

        mock_api.side_effect = on_api
        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_api.call_count, 2)
        self.assertEqual(order, ["test-1", "test-2"])
        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
        )
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
        self.assertIsNone(task_state.locked_until)

    def test_get_skips_when_task_is_backed_off(self, mock_api):
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            next_fetch_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "backoff")
        mock_api.assert_not_called()

    def test_get_skips_when_task_is_locked(self, mock_api):
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
            locked_until=datetime(2099, 1, 1, tzinfo=UTC),
        )

        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "locked")
        mock_api.assert_not_called()

    def test_get_marks_error_when_collection_fails(self, mock_api):
        mock_api.side_effect = [None, RuntimeError("boom")]

        with self.assertLogs("django.request", level="ERROR"):
            with self.assertRaises(RuntimeError):
                self.client.get(self.get_url())

        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
        )
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNone(task_state.locked_until)

    def test_get_does_not_mark_error_for_archive_download_failure(self, mock_api):
        mock_api.side_effect = ArchiveDownloadError("archive boom")

        with self.assertLogs("django.request", level="ERROR"):
            with self.assertRaises(ArchiveDownloadError):
                self.client.get(self.get_url())

        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=Exchange.COINBASE,
        )
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
        self.assertIsNone(task_state.locked_until)
