from uuid import uuid4

from django.db.models.signals import post_delete
from django.dispatch import receiver

from quant_tick.constants import FileData
from quant_tick.models import TradeData


@receiver(post_delete, sender=TradeData, dispatch_uid=uuid4())
def post_delete_file_data(sender: type[TradeData], **kwargs) -> None:
    """Post delete file data."""
    instance = kwargs["instance"]
    for file_data in FileData:
        getattr(instance, file_data).delete(save=False)
