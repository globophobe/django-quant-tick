from django.db import models

try:
    from django.utils.translation import gettext_lazy as _  # Django >= 4
except ImportError:
    from django.utils.translation import ugettext_lazy as _  # noqa


class Exchange(models.TextChoices):
    # ALPACA = "alpaca", "Alpaca"
    BINANCE = "binance", "Binance"
    BITFINEX = "bitfinex", "Bitfinex"
    # BITFLYER = "bitflyer", "bitFlyer"
    BITMEX = "bitmex", "BitMEX"
    BYBIT = "bybit", "Bybit"
    COINBASE = "coinbase", "Coinbase"
    # DERIBIT = "deribit", "Deribit"
    FTX = "ftx", "FTX"
    # UPBIT = "upbit", "Upbit"


class SamplingType(models.TextChoices):
    TIME = "time", _("time").capitalize()
    VOLUME = "volume", _("volume").capitalize()
    NOTIONAL = "notional", _("notional").capitalize()
    TICKS = "ticks", _("ticks").capitalize()
