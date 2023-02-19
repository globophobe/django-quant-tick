from uuid import uuid4

from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver

from quant_candles.models import Candle, CandleReadOnlyData


@receiver(post_delete, sender=Candle, dispatch_uid=uuid4())
def post_delete_candle(sender, **kwargs):
    """Post delete candle.

    CandleReadOnlyData is saved to another database, so will not CASCADE.
    Delete objects manually.
    """
    candle = kwargs["instance"]
    if settings.IS_LOCAL:
        CandleReadOnlyData.objects.filter(candle_id=candle.id).delete()
