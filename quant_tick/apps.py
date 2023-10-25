from django.apps import AppConfig


class QuantTickConfig(AppConfig):
    name = "quant_tick"
    verbose_name = "Quant Tick"

    def ready(self):
        import quant_tick.signals  # noqa F401
