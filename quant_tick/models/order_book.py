from decimal import Decimal, localcontext
from pathlib import PurePosixPath
from uuid import uuid4

import pandas as pd
from django.db import models

from quant_tick.constants import NUMERIC_PRECISION, NUMERIC_SCALE, Frequency

from .base import AbstractDataStorage


def order_book_upload_to(instance, filename):
    return str(
        PurePosixPath(
            "order_book",
            str(instance.symbol_id),
            instance.timestamp.strftime("%Y/%m/%d"),
            uuid4().hex + ".parquet",
        )
    )


class OrderBookDataQuerySet(models.QuerySet):
    def overlapping(self, symbol, timestamp_from=None, timestamp_to=None):
        queryset = self.filter(symbol=symbol)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        if timestamp_from is not None:
            intervals = models.Q()
            for frequency in (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY):
                intervals |= models.Q(
                    frequency=frequency,
                    timestamp__gt=timestamp_from - pd.Timedelta(minutes=frequency),
                )
            queryset = queryset.filter(intervals)
        return queryset


class OrderBookData(AbstractDataStorage):
    """Completed UTC minute summaries of displayed depth and executed flow.

    Quantities use base units (notional); volume uses quote units. Summary
    definitions and replay ordering belong to exchanges.order_book.replay.
    Minute, hour, and day Parquet files retain individual minutes with the
    standard 38 decimal places. Frequency describes storage, not sampling.
    """

    symbol = models.ForeignKey(
        "quant_tick.Symbol", on_delete=models.CASCADE, related_name="order_book_data"
    )
    timestamp = models.DateTimeField(db_index=True)
    frequency = models.PositiveIntegerField(
        choices=[
            (value, value.label)
            for value in (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY)
        ],
        default=Frequency.MINUTE,
        db_index=True,
    )
    file_data = models.FileField(max_length=255, upload_to=order_book_upload_to)

    objects = OrderBookDataQuerySet.as_manager()

    @classmethod
    def prepare_data(cls, data_frame):
        frame = data_frame.copy()
        with localcontext() as context:
            context.prec = NUMERIC_PRECISION
            scale = Decimal(1).scaleb(-NUMERIC_SCALE)
            for field in frame.columns.difference(["timestamp"]):
                frame[field] = frame[field].map(
                    lambda value: Decimal(value).quantize(scale)
                )
        return super().prepare_data(frame)

    def get_data_frame(self, field="file_data"):
        if not self.file_data.name:
            raise ValueError("Order book partition has no stored summaries")
        return super().get_data_frame(field)

    class Meta:
        db_table = "quant_tick_order_book_data"
        unique_together = (("symbol", "timestamp"),)
        ordering = ("timestamp",)
