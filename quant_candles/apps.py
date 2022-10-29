from django.apps import AppConfig


class QuantCandlesConfig(AppConfig):
    name = "quant_candles"
    verbose_name = "Quant Candles"

    def ready(self):
        import quant_candles.signals  # noqa F401
