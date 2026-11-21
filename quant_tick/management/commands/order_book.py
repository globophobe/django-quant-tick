from django.core.management.base import CommandError

from quant_tick.constants import Exchange, TaskType
from quant_tick.exchanges.order_book.collector import order_book
from quant_tick.management.base import BaseTradeDataWithRetryCommand
from quant_tick.models import Symbol, TaskState
from quant_tick.services.task_lease import TaskLeaseHeartbeat


class Command(BaseTradeDataWithRetryCommand):
    help = "Collect order book minute summaries forward from saved snapshots."

    def get_queryset(self):
        return Symbol.objects.filter(
            exchange=Exchange.HYPERLIQUID, collect_order_book=True
        )

    def handle(self, *args, **options):
        for kwargs in super().handle(*args, **options):
            symbol = kwargs["symbol"]
            state, _ = TaskState.objects.get_or_create(
                task_type=TaskType.FETCH_EXCHANGE_DATA,
                exchange=symbol.exchange,
                api_symbol=symbol.api_symbol,
            )
            if not state.acquire():
                raise CommandError(
                    "Exchange data collection is already running for this symbol"
                )
            heartbeat = TaskLeaseHeartbeat(state=state)
            try:
                heartbeat.start()
                order_book(**kwargs, assert_lease_owned=heartbeat.assert_owned)
            finally:
                heartbeat.stop()
                state.release()
