from django.db.models.signals import post_delete
from django.dispatch import receiver

from quant_candles.models import TradeData


@receiver(post_delete, sender=TradeData)
def post_delete_trade_data(sender, **kwargs):
    """Post delete trade data."""
    trades = kwargs["instance"]
    # Clean up
    trades.file_data.delete(save=False)
