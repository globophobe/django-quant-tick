from datetime import datetime

from django.db import models, transaction
from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.constants import SymbolType
from quant_tick.lib import (
    normalize_timestamp_data_frame,
    to_decimal_or_none,
    to_utc_datetime,
)

from .base import (
    BigDecimalField,
    JSONField,
    get_model_json_data,
)
from .symbols import Symbol


class FundingDataQuerySet(QuerySet):
    """QuerySet helpers for funding data."""

    def in_range(
        self,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> QuerySet:
        return self.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )


class FundingData(models.Model):
    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        related_name="funding_data",
        on_delete=models.CASCADE,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    funding_rate = BigDecimalField(_("funding rate"))
    json_data = JSONField(_("json data"), default=dict)
    objects = FundingDataQuerySet.as_manager()

    @classmethod
    def write(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
    ) -> None:
        """Replace funding data for half-open timestamp range."""
        if symbol.symbol_type != SymbolType.PERPETUAL:
            raise ValueError("FundingData is only for perpetuals.")

        rows = []
        frame = normalize_timestamp_data_frame(data_frame)
        for row in frame.to_dict("records"):
            timestamp = to_utc_datetime(row["timestamp"])
            if not timestamp_from <= timestamp < timestamp_to:
                continue
            json_data = get_model_json_data(
                row,
                {
                    "timestamp",
                    "funding_rate",
                },
            )
            rows.append(
                cls(
                    symbol=symbol,
                    timestamp=timestamp,
                    funding_rate=to_decimal_or_none(row["funding_rate"]),
                    json_data=json_data,
                )
            )

        with transaction.atomic():
            cls.objects.in_range(symbol, timestamp_from, timestamp_to).delete()
            cls.objects.bulk_create(rows)

    def to_row(self) -> dict:
        row = {
            "timestamp": self.timestamp,
            "funding_rate": self.funding_rate,
        }
        if self.json_data:
            row.update(self.json_data)
        return row

    class Meta:
        db_table = "quant_tick_funding_data"
        ordering = ("timestamp",)
        unique_together = (("symbol", "timestamp"),)
        verbose_name = verbose_name_plural = _("funding data")
