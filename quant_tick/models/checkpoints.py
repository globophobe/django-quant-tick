from pathlib import PurePosixPath
from uuid import uuid4

from django.db import models

from quant_tick.constants import CheckpointType


def checkpoint_upload_to(instance, filename):
    return str(
        PurePosixPath(
            "checkpoints",
            str(instance.symbol_id),
            instance.checkpoint_type,
            uuid4().hex + ".json.gz",
        )
    )


class Checkpoint(models.Model):
    """Persist collector state at a time boundary, isolated by symbol and type.

    Each collector owns its state format and validation.
    """

    symbol = models.ForeignKey("quant_tick.Symbol", on_delete=models.CASCADE)
    checkpoint_type = models.CharField(max_length=255, choices=CheckpointType.choices)
    timestamp = models.DateTimeField(db_index=True)
    json_data = models.JSONField()
    file_data = models.FileField(
        blank=True, max_length=255, upload_to=checkpoint_upload_to
    )

    class Meta:
        db_table = "quant_tick_checkpoint"
        unique_together = (("symbol", "checkpoint_type", "timestamp"),)
