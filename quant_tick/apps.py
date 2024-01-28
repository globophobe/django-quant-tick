from django.apps import AppConfig


class QuantTickConfig(AppConfig):
    """Quant Tick Config."""

    name = "quant_tick"
    verbose_name = "Quant Tick"

    def ready(self) -> None:
        """Ready."""
        import quant_tick.signals  # noqa F401
