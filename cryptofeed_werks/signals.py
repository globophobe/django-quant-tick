from django.db.models.signals import post_delete
from django.dispatch import receiver

from cryptofeed_werks.models import Candle


@receiver(post_delete, sender=Candle)
def post_delete_candle(sender, **kwargs):
    candle = kwargs["instance"]
    candle.aggregated_trades.all().delete()
