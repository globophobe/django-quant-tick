from django.db import InterfaceError, OperationalError

TRANSIENT_TASK_ERRORS = (OperationalError, InterfaceError)


def is_transient_task_error(exc: Exception) -> bool:
    """Return whether a task failure should be retried without backoff."""
    return isinstance(exc, TRANSIENT_TASK_ERRORS)
