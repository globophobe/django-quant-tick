from datetime import UTC, datetime

import time_machine
from django.test import TestCase, override_settings

from quant_tick.constants import TaskType
from quant_tick.models import TaskState
from quant_tick.services.task_lease import TaskLeaseLost, renew_task_lease


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
            api_symbol="BTC-USD",
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
            api_symbol="BTC-USD",
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
    def test_mark_recent_error_without_backoff_does_not_delay_next_run(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            api_symbol="BTC-USD",
        )

        task_state.mark_recent_error(backoff=False)
        task_state.refresh_from_db()

        self.assertEqual(task_state.recent_error_count, 1)
        self.assertIsNone(task_state.next_fetch_at)

        task_state.mark_recent_error(now=datetime(2024, 1, 1, 12, 10, tzinfo=UTC))
        task_state.refresh_from_db()

        self.assertEqual(task_state.recent_error_count, 1)
        self.assertEqual(
            task_state.next_fetch_at,
            datetime(2024, 1, 1, 12, 20, tzinfo=UTC),
        )

    @time_machine.travel(datetime(2024, 1, 1, 12, 0, tzinfo=UTC), tick=False)
    def test_acquire_sets_lock_lease(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            api_symbol="BTC-USD",
        )

        acquired = task_state.acquire()
        task_state.refresh_from_db()

        self.assertTrue(acquired)
        self.assertEqual(
            task_state.locked_until,
            datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        )
        self.assertIsNotNone(task_state.lock_token)

    @time_machine.travel(datetime(2024, 1, 1, 12, 1, tzinfo=UTC), tick=False)
    def test_acquire_fails_while_lease_is_active(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            api_symbol="BTC-USD",
            locked_until=datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
        )

        self.assertFalse(task_state.acquire())

    def test_heartbeat_renews_only_the_current_owner(self):
        acquired_at = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            api_symbol="BTC-USD",
        )
        self.assertTrue(task_state.acquire(now=acquired_at))
        token = task_state.lock_token

        with time_machine.travel(
            datetime(2024, 1, 1, 12, 5, tzinfo=UTC),
            tick=False,
        ):
            renew_task_lease(state=task_state)

        task_state.refresh_from_db()
        self.assertEqual(task_state.lock_token, token)
        self.assertEqual(
            task_state.locked_until,
            datetime(2024, 1, 1, 12, 15, tzinfo=UTC),
        )

    def test_expired_worker_cannot_renew_or_release_new_owner(self):
        acquired_at = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        first_owner = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            api_symbol="BTC-USD",
        )
        self.assertTrue(first_owner.acquire(now=acquired_at))
        first_token = first_owner.lock_token
        TaskState.objects.filter(pk=first_owner.pk).update(
            locked_until=datetime(2024, 1, 1, 12, 9, tzinfo=UTC),
        )

        second_owner = TaskState.objects.get(pk=first_owner.pk)
        self.assertTrue(
            second_owner.acquire(
                now=datetime(2024, 1, 1, 12, 10, tzinfo=UTC),
            )
        )
        second_token = second_owner.lock_token
        second_locked_until = second_owner.locked_until
        self.assertNotEqual(second_token, first_token)

        with self.assertRaisesRegex(TaskLeaseLost, "ownership lost"):
            renew_task_lease(state=first_owner)
        first_owner.release()

        second_owner.refresh_from_db()
        self.assertEqual(second_owner.lock_token, second_token)
        self.assertEqual(second_owner.locked_until, second_locked_until)

        second_owner.release()
        second_owner.refresh_from_db()
        self.assertIsNone(second_owner.lock_token)
        self.assertIsNone(second_owner.locked_until)

    def test_clear_recent_error_resets_backoff_state(self):
        task_state = TaskState.objects.create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange="coinbase",
            api_symbol="BTC-USD",
            recent_error_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            recent_error_count=3,
            next_fetch_at=datetime(2024, 1, 1, 12, 20, tzinfo=UTC),
        )

        task_state.clear_recent_error()
        task_state.refresh_from_db()

        self.assertIsNone(task_state.recent_error_at)
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
