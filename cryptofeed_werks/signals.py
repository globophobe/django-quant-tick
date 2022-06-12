from django.db.models.signals import post_delete
from django.dispatch import receiver

from cryptofeed_werks.models import AggregatedTradeData


@receiver(post_delete, sender=AggregatedTradeData)
def post_delete_aggregated_trade_data(sender, **kwargs):
    aggregated_trade = kwargs["instance"]
    # Clean up
    aggregated_trade.data.delete(save=False)
