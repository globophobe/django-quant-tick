from datetime import datetime, timedelta

from django.conf import settings
from django.db import models
from django.db.models import Q
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from quant_tick.constants import Exchange, TaskType

QUANT_TICK_TASK_BACKOFF_BASE = timedelta(minutes=10)
QUANT_TICK_TASK_BACKOFF_CAP = timedelta(hours=1)
QUANT_TICK_TASK_BACKOFF_MULTIPLIER = 2
QUANT_TICK_TASK_LOCK_LEASE = timedelta(hours=1)


def _as_timedelta(value: int | float | timedelta | None, default: timedelta) -> timedelta:
    """Coerce a setting value to timedelta."""
    if value is None:
        return default
    if isinstance(value, timedelta):
        return value
    return timedelta(seconds=int(value))


def get_task_backoff(count: int) -> timedelta:
    """Get exponential task backoff for a consecutive error count."""
    if count <= 0:
        return timedelta(0)
    base = _as_timedelta(
        getattr(settings, "QUANT_TICK_TASK_BACKOFF_BASE_SECONDS", None),
        QUANT_TICK_TASK_BACKOFF_BASE,
    )
    cap = _as_timedelta(
        getattr(settings, "QUANT_TICK_TASK_BACKOFF_CAP_SECONDS", None),
        QUANT_TICK_TASK_BACKOFF_CAP,
    )
    multiplier = max(
        1,
        int(
            getattr(
                settings,
                "QUANT_TICK_TASK_BACKOFF_MULTIPLIER",
                QUANT_TICK_TASK_BACKOFF_MULTIPLIER,
            )
        ),
    )
    backoff = base
    for _idx in range(1, count):
        backoff *= multiplier
        if backoff >= cap:
            return cap
    return min(backoff, cap)


def get_task_lock_lease() -> timedelta:
    """Get the lock lease duration for a task run."""
    return _as_timedelta(
        getattr(settings, "QUANT_TICK_TASK_LOCK_LEASE_SECONDS", None),
        QUANT_TICK_TASK_LOCK_LEASE,
    )


class TaskState(models.Model):
    """Per-task lock and backoff state."""

    exchange = models.CharField(
        _("exchange"),
        choices=Exchange.choices,
        max_length=255,
        blank=True,
        default="",
    )
    task_type = models.CharField(
        _("task type"),
        choices=TaskType.choices,
        max_length=255,
    )
    recent_error_at = models.DateTimeField(
        _("recent error"),
        help_text=_("Last task error time used for observability."),
        null=True,
        blank=True,
    )
    recent_error_count = models.PositiveIntegerField(
        _("recent error count"),
        help_text=_("Consecutive task errors used to determine exponential backoff."),
        default=0,
    )
    next_fetch_at = models.DateTimeField(
        _("next fetch at"),
        help_text=_("Do not run this task before this time if backoff is active."),
        null=True,
        blank=True,
    )
    locked_until = models.DateTimeField(
        _("locked until"),
        help_text=_("Lease expiry for preventing overlapping task runs."),
        null=True,
        blank=True,
    )

    def can_run(self, *, now: datetime | None = None) -> bool:
        """Whether the task is outside its backoff window."""
        current_time = now or timezone.now()
        return self.next_fetch_at is None or self.next_fetch_at <= current_time

    def acquire(self, *, now: datetime | None = None) -> bool:
        """Acquire the task lease if it is not already held."""
        current_time = now or timezone.now()
        locked_until = current_time + get_task_lock_lease()
        updated = TaskState.objects.filter(pk=self.pk).filter(
            Q(locked_until__isnull=True) | Q(locked_until__lte=current_time)
        ).update(locked_until=locked_until)
        if not updated:
            return False
        self.locked_until = locked_until
        return True

    def release(self) -> None:
        """Release the task lease."""
        if self.locked_until is None:
            return
        self.locked_until = None
        self.save(update_fields=["locked_until"])

    def mark_recent_error(self, *, now: datetime | None = None) -> None:
        """Mark a task failure and advance its backoff state."""
        current_time = now or timezone.now()
        self.recent_error_at = current_time
        self.recent_error_count += 1
        self.next_fetch_at = current_time + get_task_backoff(self.recent_error_count)
        self.save(
            update_fields=["recent_error_at", "recent_error_count", "next_fetch_at"]
        )

    def clear_recent_error(self) -> None:
        """Clear the task backoff state after a successful run."""
        if (
            self.recent_error_at is None
            and self.recent_error_count == 0
            and self.next_fetch_at is None
        ):
            return
        self.recent_error_at = None
        self.recent_error_count = 0
        self.next_fetch_at = None
        self.save(
            update_fields=["recent_error_at", "recent_error_count", "next_fetch_at"]
        )

    class Meta:
        db_table = "quant_tick_task_state"
        constraints = [
            models.UniqueConstraint(
                fields=("task_type", "exchange"),
                name="quant_tick_task_state_unique",
            )
        ]
        ordering = ("task_type", "exchange")
        verbose_name = _("task state")
        verbose_name_plural = _("task states")
