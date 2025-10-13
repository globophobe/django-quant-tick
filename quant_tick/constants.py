from decimal import Decimal

from django.db import models

from quant_tick.utils import gettext_lazy as _

ZERO = Decimal("0")
ONE = Decimal("1")


# Similar to BigQuery BigNumeric
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
NUMERIC_PRECISION = 76  # 76.6
NUMERIC_SCALE = 38


class Exchange(models.TextChoices):
    """Exchange."""

    BINANCE = "binance", "Binance"
    BITFINEX = "bitfinex", "Bitfinex"
    BITMEX = "bitmex", "BitMEX"
    COINBASE = "coinbase", "Coinbase"
    DRIFT = "drift", "Drift"


class SymbolType(models.TextChoices):
    """Symbol type."""

    SPOT = "spot", _("spot")
    PERP = "perp", _("perp")
    FUTURE = "future", _("future")


class SampleType(models.TextChoices):
    """Sample type."""

    VOLUME = "volume", _("volume")
    NOTIONAL = "notional", _("notional")
    TICK = "ticks", _("tick")  # Pluralized to match data frame.


class Frequency(models.IntegerChoices):
    """Frequency."""

    MINUTE = 1, _("minute").capitalize()
    HOUR = 60, _("hour").capitalize()
    DAY = 1440, _("day").capitalize()
    WEEK = 10080, _("week").capitalize()


class PositionType(models.TextChoices):
    """Position type."""

    BACKTEST = "backtest", _("backtest")
    PAPER = "paper", _("paper")
    LIVE = "live", _("live")


class ExitReason(models.TextChoices):
    """Exit reason."""

    TAKE_PROFIT = "take_profit", _("take profit")
    STOP_LOSS = "stop_loss", _("stop loss")
    MAX_DURATION = "max_duration", _("max duration")


class PositionStatus(models.TextChoices):
    """Position status."""

    PENDING = "pending", _("pending")
    OPEN = "open", _("open")
    CLOSED = "closed", _("closed")
    FAILED = "failed", _("failed")


class FileData(models.TextChoices):
    """File data."""

    RAW = "raw_data", _("raw")
    AGGREGATED = "aggregated_data", _("aggregated")
    FILTERED = "filtered_data", _("filtered")
    CANDLE = "candle_data", _("candles")
