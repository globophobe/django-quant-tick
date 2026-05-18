from datetime import UTC, datetime
from unittest.mock import patch

from django.test import TestCase

from quant_tick.constants import Exchange, TaskType
from quant_tick.models import Candle, Symbol, TaskState
from quant_tick.services.aggregate_candles import (
    DEFAULT_TIMESTAMP_FROM,
    aggregate_candle_data,
)


class AggregateCandleServiceTest(TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )

    def test_aggregate_processes_all_candles(self):
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
            response = aggregate_candle_data([{}])

        self.assertEqual(response, {"ok": True, "processed": 2})
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

    def test_aggregate_scopes_by_exchange(self):
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
            response = aggregate_candle_data([{"exchange": Exchange.COINBASE}])

        self.assertEqual(response, {"ok": True, "processed": 1})
        self.assertEqual(mock_candles.call_count, 1)
        self.assertEqual(order, [coinbase.code_name])

    def test_aggregate_scopes_by_api_symbol(self):
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
            response = aggregate_candle_data(
                [{"exchange": Exchange.COINBASE, "api_symbol": "test"}]
            )

        self.assertEqual(response, {"ok": True, "processed": 1})
        self.assertEqual(mock_candles.call_count, 1)
        self.assertEqual(order, [coinbase.code_name])

    def test_aggregate_defaults_to_tail_forward_start(self):
        Candle.objects.create(symbol=self.symbol)
        now = datetime(2026, 5, 9, 12, 30, 45, tzinfo=UTC)

        with patch(
            "quant_tick.services.aggregate_candles.get_current_time",
            return_value=now,
        ):
            with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
                response = aggregate_candle_data(
                    [{"exchange": Exchange.COINBASE, "api_symbol": "test"}]
                )

        self.assertEqual(response, {"ok": True, "processed": 1})
        timestamp_from, timestamp_to, retry = mock_candles.call_args.args
        self.assertEqual(timestamp_from, DEFAULT_TIMESTAMP_FROM)
        self.assertEqual(timestamp_to, datetime(2026, 5, 9, 12, 30, tzinfo=UTC))
        self.assertFalse(retry)

    def test_aggregate_retries_from_explicit_timestamp(self):
        Candle.objects.create(symbol=self.symbol)
        now = datetime(2026, 5, 9, 12, 30, 45, tzinfo=UTC)

        with patch(
            "quant_tick.services.aggregate_candles.get_current_time",
            return_value=now,
        ):
            with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
                response = aggregate_candle_data(
                    [
                        {
                            "exchange": Exchange.COINBASE,
                            "api_symbol": "test",
                            "timestamp_from": datetime(2026, 5, 9, 11, 42, tzinfo=UTC),
                        }
                    ]
                )

        self.assertEqual(response, {"ok": True, "processed": 1})
        timestamp_from, timestamp_to, retry = mock_candles.call_args.args
        self.assertEqual(timestamp_from, datetime(2026, 5, 9, 11, 0, tzinfo=UTC))
        self.assertEqual(timestamp_to, datetime(2026, 5, 9, 12, 30, tzinfo=UTC))
        self.assertTrue(retry)

    def test_aggregate_processes_multiple_payloads(self):
        Candle.objects.create(symbol=self.symbol)
        binance_symbol = Symbol.objects.create(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
        )
        Candle.objects.create(symbol=binance_symbol)
        now = datetime(2026, 5, 9, 12, 30, 45, tzinfo=UTC)

        with patch(
            "quant_tick.services.aggregate_candles.get_current_time",
            return_value=now,
        ):
            with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
                response = aggregate_candle_data(
                    [
                        {
                            "exchange": Exchange.COINBASE,
                            "api_symbol": "test",
                            "timestamp_from": datetime(2026, 5, 9, 11, tzinfo=UTC),
                        },
                        {
                            "exchange": Exchange.BINANCE,
                            "api_symbol": "BTCUSDT",
                        },
                    ]
                )

        self.assertEqual(response, {"ok": True, "processed": 2})
        self.assertEqual(mock_candles.call_count, 2)

    def test_aggregate_skips_when_task_is_backed_off(self):
        Candle.objects.create(symbol=self.symbol)
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=Exchange.COINBASE,
            api_symbol="test",
            next_fetch_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

        with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
            response = aggregate_candle_data(
                [{"exchange": Exchange.COINBASE, "api_symbol": "test"}]
            )

        self.assertEqual(response, {"ok": True, "skipped": "backoff"})
        mock_candles.assert_not_called()

    def test_aggregate_skips_when_task_is_locked(self):
        Candle.objects.create(symbol=self.symbol)
        TaskState.objects.create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=Exchange.COINBASE,
            api_symbol="test",
            locked_until=datetime(2099, 1, 1, tzinfo=UTC),
        )

        with patch("quant_tick.models.candles.Candle.candles") as mock_candles:
            response = aggregate_candle_data(
                [{"exchange": Exchange.COINBASE, "api_symbol": "test"}]
            )

        self.assertEqual(response, {"ok": True, "skipped": "locked"})
        mock_candles.assert_not_called()

    def test_aggregate_marks_error_when_aggregation_fails(self):
        Candle.objects.create(symbol=self.symbol)
        Candle.objects.create(symbol=self.symbol)

        with patch(
            "quant_tick.models.candles.Candle.candles",
            autospec=True,
            side_effect=[None, RuntimeError("boom")],
        ):
            with self.assertRaises(RuntimeError):
                aggregate_candle_data([{}])

        task_state = TaskState.objects.get(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNone(task_state.locked_until)
