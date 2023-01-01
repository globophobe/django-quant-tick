from uuid import uuid4

from django.db.models.signals import post_delete
from django.dispatch import receiver

from quant_candles.models import TradeData


@receiver(post_delete, sender=TradeData, dispatch_uid=uuid4())
def post_delete_trade_data(sender, **kwargs):
    """Post delete trade data."""
    # Clean up
    trade_data = kwargs["instance"]
    trade_data.file_data.delete(save=False)
