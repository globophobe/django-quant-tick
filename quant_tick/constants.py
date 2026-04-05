from decimal import Decimal

from django.db import models
from django.utils.translation import gettext_lazy as _

ZERO = Decimal("0")
ONE = Decimal("1")


# Similar to BigQuery BigNumeric
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
NUMERIC_PRECISION = 76  # 76.6
NUMERIC_SCALE = 38


class Exchange(models.TextChoices):
    """Supported exchange identifiers."""

    BINANCE = "binance", "Binance"
    BITFINEX = "bitfinex", "Bitfinex"
    BITMEX = "bitmex", "BitMEX"
    COINBASE = "coinbase", "Coinbase"


class TaskType(models.TextChoices):
    """Kinds of scheduled tasks tracked in TaskState."""

    AGGREGATE_TRADES = "aggregate_trades", "Aggregate trades"
    AGGREGATE_CANDLES = "aggregate_candles", "Aggregate candles"


class SampleType(models.TextChoices):
    """Sampling inputs for non-time-based candles."""

    VOLUME = "volume", _("volume")
    NOTIONAL = "notional", _("notional")
    TICK = "ticks", _("tick")  # Pluralized to match data frame.


class Frequency(models.IntegerChoices):
    """Trade-data frequencies in minutes."""

    MINUTE = 1, _("minute").capitalize()
    HOUR = 60, _("hour").capitalize()
    DAY = 1440, _("day").capitalize()
    WEEK = 10080, _("week").capitalize()


class FileData(models.TextChoices):
    """TradeData parquet file fields."""

    RAW = "raw_data", _("raw")
    AGGREGATED = "aggregated_data", _("aggregated")
    FILTERED = "filtered_data", _("filtered")
    CANDLE = "candle_data", _("candles")
