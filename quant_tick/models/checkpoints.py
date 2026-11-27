import gzip
import hashlib
import json
from pathlib import PurePosixPath
from uuid import uuid4

from django.db import models

from quant_tick.constants import CheckpointType
from quant_tick.testing import is_test


def checkpoint_upload_to(instance, filename):
    return str(
        PurePosixPath(
            "test-checkpoints" if is_test() else "checkpoints",
            *instance.symbol.upload_path,
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

    def read_state(self):
        """Read verified file contents, or an existing inline checkpoint."""
        if not self.file_data:
            return self.json_data
        with self.file_data.open("rb") as source:
            raw = gzip.decompress(source.read())
        if hashlib.sha256(raw).hexdigest() != self.json_data.get("sha256"):
            raise ValueError("Checkpoint file checksum mismatch")
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise TypeError("Checkpoint file must contain an object")
        return state

    class Meta:
        db_table = "quant_tick_checkpoint"
        unique_together = (("symbol", "checkpoint_type", "timestamp"),)
