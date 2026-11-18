from django.db import models


class PhoenixPositionCheckpoint(models.Model):
    """Validated position state before a UTC boundary, separate from market data."""

    symbol = models.ForeignKey("quant_tick.Symbol", on_delete=models.CASCADE)
    timestamp = models.DateTimeField(db_index=True)
    state = models.JSONField()

    class Meta:
        db_table = "quant_tick_phoenix_position_checkpoint"
        unique_together = (("symbol", "timestamp"),)
