from datetime import UTC, datetime

import time_machine
from django.test import TestCase, override_settings

from quant_tick.constants import TaskType
from quant_tick.models import TaskState


@override_settings(
    QUANT_TICK_TASK_BACKOFF_BASE_SECONDS=600,
    QUANT_TICK_TASK_BACKOFF_CAP_SECONDS=1800,
    QUANT_TICK_TASK_BACKOFF_MULTIPLIER=2,
    QUANT_TICK_TASK_LOCK_LEASE_SECONDS=600,
)
class TaskStateTest(TestCase):
    @time_machine.travel(datetime(2024, 1, 1, 12, 0, tzinfo=UTC), tick=False)
    def test_mark_recent_error_sets_first_backoff_window(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
        )

        task_state.mark_recent_error()
        task_state.refresh_from_db()

        self.assertEqual(
            task_state.recent_error_at,
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        self.assertEqual(task_state.recent_error_count, 1)
        self.assertEqual(
            task_state.next_fetch_at,
            datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        )

    @time_machine.travel(datetime(2024, 1, 1, 12, 10, tzinfo=UTC), tick=False)
    def test_mark_recent_error_grows_backoff_exponentially(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            recent_error_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            recent_error_count=1,
            next_fetch_at=datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        )

        task_state.mark_recent_error()
        task_state.refresh_from_db()

        self.assertEqual(task_state.recent_error_count, 2)
        self.assertEqual(
            task_state.next_fetch_at,
            datetime(2024, 1, 1, 12, 30, tzinfo=UTC),
        )

    @time_machine.travel(datetime(2024, 1, 1, 12, 0, tzinfo=UTC), tick=False)
    def test_acquire_sets_lock_lease(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
        )

        acquired = task_state.acquire()
        task_state.refresh_from_db()

        self.assertTrue(acquired)
        self.assertEqual(
            task_state.locked_until,
            datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        )

    @time_machine.travel(datetime(2024, 1, 1, 12, 1, tzinfo=UTC), tick=False)
    def test_acquire_fails_while_lease_is_active(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            locked_until=datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        )

        self.assertFalse(task_state.acquire())

    def test_clear_recent_error_resets_backoff_state(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            recent_error_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            recent_error_count=3,
            next_fetch_at=datetime(2024, 1, 1, 12, 20, tzinfo=UTC),
        )

        task_state.clear_recent_error()
        task_state.refresh_from_db()

        self.assertIsNone(task_state.recent_error_at)
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
