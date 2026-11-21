import logging

from django.db import transaction
from django.db.models.signals import post_delete, pre_delete
from django.dispatch import receiver

from quant_tick.models import Checkpoint, OrderBookData

logger = logging.getLogger(__name__)


def discard(storage, name):
    try:
        storage.delete(name)
    except Exception:
        logger.exception("Stored file cleanup failed: %s", name)


@receiver(pre_delete, sender=Checkpoint, dispatch_uid="checkpoint_pre_delete")
@receiver(pre_delete, sender=OrderBookData, dispatch_uid="order_book_pre_delete")
def remember_file(sender, instance, **kwargs):
    if not instance.file_data.name:
        return
    instance._deleted_file = instance.file_data.storage, instance.file_data.name


@receiver(post_delete, sender=Checkpoint, dispatch_uid="checkpoint_post_delete")
@receiver(post_delete, sender=OrderBookData, dispatch_uid="order_book_post_delete")
def delete_file(sender, instance, **kwargs):
    if not hasattr(instance, "_deleted_file"):
        return
    storage, name = instance._deleted_file

    def remove():
        discard(storage, name)

    # Rolling back a row deletion must leave its referenced file readable.
    transaction.on_commit(remove, using=kwargs.get("using"))
