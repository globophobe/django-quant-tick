from django.apps import AppConfig


class QuantWerksConfig(AppConfig):
    name = "quant_werks"
    verbose_name = "Quant Werks"

    def ready(self):
        import quant_werks.signals  # noqa F401
