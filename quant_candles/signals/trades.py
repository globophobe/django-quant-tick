from uuid import uuid4

from django.db.models.signals import post_delete
from django.dispatch import receiver

from quant_candles.constants import Frequency
from quant_candles.models import TradeData, TradeDataSummary


@receiver(post_delete, sender=TradeData, dispatch_uid=uuid4())
def post_delete_hourly_trade_data(sender, **kwargs):
    """Post delete hourly trade data.

    * Exclude minute data, which will be deleted if converted to hourly.
    """
    instance = kwargs["instance"]
    if instance.frequency == Frequency.HOUR.value:
        trade_data_summary = TradeDataSummary.objects.filter(
            symbol=instance.symbol, date=instance.timestamp.date()
        )
        trade_data_summary.delete()


@receiver(post_delete, dispatch_uid=uuid4())
def post_delete_file_data(sender, **kwargs):
    """Post delete file data."""
    # Clean up
    if sender in (TradeData, TradeDataSummary):
        instance = kwargs["instance"]
        instance.file_data.delete(save=False)
