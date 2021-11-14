from django.apps import AppConfig


class CryptofeedWerksConfig(AppConfig):
    name = "cryptofeed_werks"
    verbose_name = "Cryptofeed Werks"

    def ready(self):
        import cryptofeed_werks.signals  # noqa F401
