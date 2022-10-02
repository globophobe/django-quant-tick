from django.db import models

from quant_werks.utils import gettext_lazy as _

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
    # DERIBIT = "deribit", "Deribit"
    FTX = "ftx", "FTX"
    # UPBIT = "upbit", "Upbit"


class SymbolType(models.TextChoices):
    SPOT = "spot", _("spot")
    PERP = "perp", _("perp")
    FUTURE = "future", _("future")


class Frequency(models.IntegerChoices):
    MINUTE = 1, _("minute").capitalize()
    HOUR = 60, _("hour").capitalize()
    DAY = 1440, _("day").capitalize()
    WEEK = 10080, _("week").capitalize()
