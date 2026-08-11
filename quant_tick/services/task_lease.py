from datetime import datetime
from threading import Event, Thread

from django.db import close_old_connections, transaction
from django.utils import timezone

from quant_tick.models import TaskState
from quant_tick.models.task_state import get_task_backoff, get_task_lock_lease


class TaskLeaseLost(RuntimeError):
    pass


def _owned_state_for_update(*, state: TaskState) -> TaskState:
    token = state.lock_token
    if token is None:
        raise TaskLeaseLost(f"task lease token missing for TaskState {state.pk}")
    current = TaskState.objects.select_for_update().get(pk=state.pk)
    if current.lock_token != token:
        raise TaskLeaseLost(f"task lease ownership lost for TaskState {state.pk}")
    return current


def assert_task_lease_owned(*, state: TaskState) -> None:
    with transaction.atomic():
        _owned_state_for_update(state=state)


def renew_task_lease(*, state: TaskState) -> None:
    token = state.lock_token
    if token is None:
        raise TaskLeaseLost(f"task lease token missing for TaskState {state.pk}")
    updated = TaskState.objects.filter(
        pk=state.pk,
        lock_token=token,
    ).update(locked_until=timezone.now() + get_task_lock_lease())
    if updated != 1:
        raise TaskLeaseLost(
            f"task lease ownership lost while renewing TaskState {state.pk}"
        )


def mark_task_recent_error(
    *,
    state: TaskState,
    now: datetime | None = None,
    backoff: bool = True,
) -> None:
    current_time = now or timezone.now()
    close_old_connections()
    with transaction.atomic():
        current = _owned_state_for_update(state=state)
        current.recent_error_at = current_time
        if backoff:
            current.recent_error_count = (
                current.recent_error_count + 1 if current.next_fetch_at else 1
            )
            current.next_fetch_at = current_time + get_task_backoff(
                current.recent_error_count
            )
        else:
            current.recent_error_count += 1
            current.next_fetch_at = None
        current.save(
            update_fields=[
                "recent_error_at",
                "recent_error_count",
                "next_fetch_at",
            ]
        )
    state.recent_error_at = current.recent_error_at
    state.recent_error_count = current.recent_error_count
    state.next_fetch_at = current.next_fetch_at


def clear_task_recent_error(*, state: TaskState) -> None:
    close_old_connections()
    with transaction.atomic():
        current = _owned_state_for_update(state=state)
        if (
            current.recent_error_at is not None
            or current.recent_error_count != 0
            or current.next_fetch_at is not None
        ):
            current.recent_error_at = None
            current.recent_error_count = 0
            current.next_fetch_at = None
            current.save(
                update_fields=[
                    "recent_error_at",
                    "recent_error_count",
                    "next_fetch_at",
                ]
            )
    state.recent_error_at = current.recent_error_at
    state.recent_error_count = current.recent_error_count
    state.next_fetch_at = current.next_fetch_at


class TaskLeaseHeartbeat:
    def __init__(self, *, state: TaskState) -> None:
        self.state = state
        ttl_seconds = max(1.0, get_task_lock_lease().total_seconds())
        self.interval_seconds = max(1.0, min(60.0, ttl_seconds / 3.0))
        self._stop = Event()
        self._lost = Event()
        self._error = ""
        self._thread = Thread(
            target=self._run,
            name=f"task-lease-{state.pk}",
            daemon=True,
        )

    def start(self) -> None:
        renew_task_lease(state=self.state)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            close_old_connections()
            try:
                renew_task_lease(state=self.state)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self._lost.set()
                return
            finally:
                close_old_connections()

    def assert_owned(self) -> None:
        if self._lost.is_set():
            raise TaskLeaseLost(
                "task lease heartbeat lost ownership for "
                f"TaskState {self.state.pk}: {self._error}"
            )
        assert_task_lease_owned(state=self.state)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
