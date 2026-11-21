import logging

from django.db import transaction
from django.db.models.signals import post_delete, pre_delete
from django.dispatch import receiver

from quant_tick.models import Checkpoint

logger = logging.getLogger(__name__)


@receiver(pre_delete, sender=Checkpoint, dispatch_uid="checkpoint_pre_delete")
def remember_checkpoint_file(sender, instance, **kwargs):
    if not instance.data.name:
        return
    instance._deleted_file = instance.data.storage, instance.data.name


@receiver(post_delete, sender=Checkpoint, dispatch_uid="checkpoint_post_delete")
def delete_checkpoint_file(sender, instance, **kwargs):
    if not hasattr(instance, "_deleted_file"):
        return
    storage, name = instance._deleted_file

    def remove():
        try:
            storage.delete(name)
        except Exception:
            logger.exception("Checkpoint file cleanup failed: %s", name)

    # Rolling back a row deletion must leave its referenced snapshot readable.
    transaction.on_commit(remove, using=kwargs.get("using"))
