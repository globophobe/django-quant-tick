from collections.abc import Callable
from datetime import datetime

import pandas as pd
from django.db import models, transaction
from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.constants import Exchange, SymbolType
from quant_tick.lib import (
    normalize_timestamp_data_frame,
    to_decimal_or_none,
    to_utc_datetime,
)

from .base import BigDecimalField, JSONField, get_model_json_data
from .symbols import Symbol

PERPETUAL_STATS_VALUE_FIELDS = (
    "open_interest",
    "open_interest_value",
    "single_open_interest",
    "top_trader_long_short_account_ratio",
    "top_trader_long_short_position_ratio",
    "long_short_account_ratio",
    "taker_long_short_volume_ratio",
    "long_account_ratio",
    "short_account_ratio",
)
PERPETUAL_STATS_REQUIRED_FIELDS = {
    Exchange.BINANCE_FUTURES: (
        "open_interest",
        "open_interest_value",
        "top_trader_long_short_account_ratio",
        "top_trader_long_short_position_ratio",
        "long_short_account_ratio",
        "taker_long_short_volume_ratio",
    ),
    Exchange.BYBIT_LINEAR: (
        "open_interest",
        "single_open_interest",
        "long_account_ratio",
        "short_account_ratio",
    ),
    Exchange.BYBIT_INVERSE: (
        "open_interest",
        "single_open_interest",
        "long_account_ratio",
        "short_account_ratio",
    ),
}


def perpetual_stats_required_fields(exchange: str) -> tuple[str, ...]:
    """Return fields proving every native endpoint contributed an observation."""
    try:
        return PERPETUAL_STATS_REQUIRED_FIELDS[Exchange(exchange)]
    except (KeyError, ValueError) as exc:
        raise NotImplementedError(
            f"Perpetual stats completeness is not defined for {exchange}."
        ) from exc


class PerpetualStatsDataQuerySet(QuerySet):
    """QuerySet helpers for native-cadence perpetual stats observations."""

    def in_range(
        self,
        symbol: Symbol,
        frequency: int,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> QuerySet:
        return self.filter(
            symbol=symbol,
            frequency=frequency,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )

    def complete_for_exchange(self, exchange: str) -> QuerySet:
        """Return observations populated by every required venue endpoint."""
        required = perpetual_stats_required_fields(exchange)
        return self.filter(**{f"{field}__isnull": False for field in required})


class PerpetualStatsData(models.Model):
    """Native-cadence open-interest and positioning observations."""

    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        related_name="perpetual_stats_data",
        on_delete=models.CASCADE,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(_("frequency"), db_index=True)
    open_interest = BigDecimalField(_("open interest"), null=True, blank=True)
    open_interest_value = BigDecimalField(
        _("open interest value"), null=True, blank=True
    )
    single_open_interest = BigDecimalField(
        _("single open interest"), null=True, blank=True
    )
    top_trader_long_short_account_ratio = BigDecimalField(
        _("top trader long short account ratio"), null=True, blank=True
    )
    top_trader_long_short_position_ratio = BigDecimalField(
        _("top trader long short position ratio"), null=True, blank=True
    )
    long_short_account_ratio = BigDecimalField(
        _("long short account ratio"), null=True, blank=True
    )
    taker_long_short_volume_ratio = BigDecimalField(
        _("taker long short volume ratio"), null=True, blank=True
    )
    long_account_ratio = BigDecimalField(
        _("long account ratio"), null=True, blank=True
    )
    short_account_ratio = BigDecimalField(
        _("short account ratio"), null=True, blank=True
    )
    open_interest_unit = models.CharField(
        _("open interest unit"), max_length=32, blank=True, default=""
    )
    json_data = JSONField(_("json data"), default=dict)
    objects = PerpetualStatsDataQuerySet.as_manager()

    @classmethod
    def write(
        cls,
        symbol: Symbol,
        frequency: int,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        *,
        assert_lease_owned: Callable[[], None] | None = None,
    ) -> None:
        """Upsert returned observations without deleting absent timestamps."""
        if symbol.symbol_type != SymbolType.PERPETUAL:
            raise ValueError("PerpetualStatsData is only for perpetuals.")
        if frequency <= 0:
            raise ValueError("PerpetualStatsData frequency must be positive.")

        rows = []
        frame = normalize_timestamp_data_frame(data_frame)
        excluded = {
            "timestamp",
            "frequency",
            "open_interest_unit",
            *PERPETUAL_STATS_VALUE_FIELDS,
        }
        for row in frame.to_dict("records"):
            timestamp = to_utc_datetime(row["timestamp"])
            if not timestamp_from <= timestamp < timestamp_to:
                continue
            values = {
                field: to_decimal_or_none(row.get(field))
                for field in PERPETUAL_STATS_VALUE_FIELDS
            }
            rows.append(
                cls(
                    symbol=symbol,
                    timestamp=timestamp,
                    frequency=frequency,
                    open_interest_unit=str(row.get("open_interest_unit") or ""),
                    json_data=get_model_json_data(row, excluded),
                    **values,
                )
            )

        with transaction.atomic():
            if assert_lease_owned is not None:
                assert_lease_owned()
            if rows:
                existing_by_timestamp = {
                    item.timestamp: item
                    for item in cls.objects.select_for_update().filter(
                        symbol=symbol,
                        frequency=frequency,
                        timestamp__in=[row.timestamp for row in rows],
                    )
                }
                required = perpetual_stats_required_fields(symbol.exchange)
                replacements = []
                for row in rows:
                    existing = existing_by_timestamp.get(row.timestamp)
                    if existing is None:
                        replacements.append(row)
                        continue
                    incoming_complete = all(
                        getattr(row, field) is not None for field in required
                    )
                    existing_complete = all(
                        getattr(existing, field) is not None for field in required
                    )
                    if existing_complete and not incoming_complete:
                        continue
                    for field in PERPETUAL_STATS_VALUE_FIELDS:
                        if getattr(row, field) is None:
                            setattr(row, field, getattr(existing, field))
                    if not row.open_interest_unit:
                        row.open_interest_unit = existing.open_interest_unit
                    row.json_data = {
                        **(existing.json_data or {}),
                        **(row.json_data or {}),
                    }
                    replacements.append(row)
                replacement_timestamps = [row.timestamp for row in replacements]
                cls.objects.filter(
                    symbol=symbol,
                    frequency=frequency,
                    timestamp__in=replacement_timestamps,
                ).delete()
                cls.objects.bulk_create(replacements, batch_size=1000)

    def coverage_end(self) -> datetime:
        return self.timestamp + pd.Timedelta(minutes=self.frequency)

    def to_row(self) -> dict:
        row = {
            "timestamp": self.timestamp,
            **{field: getattr(self, field) for field in PERPETUAL_STATS_VALUE_FIELDS},
            "open_interest_unit": self.open_interest_unit,
        }
        if self.json_data:
            row.update(self.json_data)
        return row

    class Meta:
        db_table = "quant_tick_perpetual_stats_data"
        ordering = ("timestamp",)
        unique_together = (("symbol", "frequency", "timestamp"),)
        verbose_name = verbose_name_plural = _("perpetual stats data")
