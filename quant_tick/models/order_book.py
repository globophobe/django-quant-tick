from django.db import models

from .base import JSONField


class OrderBookData(models.Model):
    """Completed UTC minute summaries of displayed depth and executed flow.

    Quantities use base units (notional); volume uses quote units. Summary
    definitions and replay ordering belong to exchanges.order_book.replay.
    """

    symbol = models.ForeignKey(
        "quant_tick.Symbol", on_delete=models.CASCADE, related_name="order_book_data"
    )
    timestamp = models.DateTimeField(db_index=True)
    json_data = JSONField("data")

    class Meta:
        db_table = "quant_tick_order_book_data"
        unique_together = (("symbol", "timestamp"),)
        ordering = ("timestamp",)
