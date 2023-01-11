from uuid import uuid4

from django.db.models.signals import post_delete
from django.dispatch import receiver

from quant_candles.models import TradeData, TradeDataSummary


@receiver(post_delete, dispatch_uid=uuid4())
def post_delete_file_data(sender, **kwargs):
    """Post delete file data."""
    # Clean up
    if sender in (TradeData, TradeDataSummary):
        instance = kwargs["instance"]
        instance.file_data.delete(save=False)
