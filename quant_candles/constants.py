from django.db import models

from quant_candles.utils import gettext_lazy as _

# Similar to BigQuery BigNumeric
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
NUMERIC_PRECISION = 76  # 76.6
NUMERIC_SCALE = 38


class Exchange(models.TextChoices):
    # ALPACA = "alpaca", "Alpaca"
    BINANCE = "binance", "Binance"
    BITFINEX = "bitfinex", "Bitfinex"
    # BITFLYER = "bitflyer", "bitFlyer"
    BITMEX = "bitmex", "BitMEX"
    BYBIT = "bybit", "Bybit"
    COINBASE = "coinbase", "Coinbase"
    # UPBIT = "upbit", "Upbit"


class SymbolType(models.TextChoices):
    SPOT = "spot", _("spot")
    PERP = "perp", _("perp")
    FUTURE = "future", _("future")


class SampleType(models.TextChoices):
    VOLUME = "volume", _("volume")
    NOTIONAL = "notional", _("notional")
    TICK = "ticks", _("tick")  # Pluralized to match data frame.


class Frequency(models.IntegerChoices):
    MINUTE = 1, _("minute").capitalize()
    HOUR = 60, _("hour").capitalize()
    DAY = 1440, _("day").capitalize()
    WEEK = 10080, _("week").capitalize()
