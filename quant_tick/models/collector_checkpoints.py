from django.db import models


class CollectorCheckpoint(models.Model):
    """Persist collector state at a time boundary, isolated by symbol and kind.

    Each collector owns its state format and validation.
    """

    symbol = models.ForeignKey("quant_tick.Symbol", on_delete=models.CASCADE)
    kind = models.CharField(max_length=255)
    timestamp = models.DateTimeField(db_index=True)
    state = models.JSONField()

    class Meta:
        db_table = "quant_tick_collector_checkpoint"
        unique_together = (("symbol", "kind", "timestamp"),)
