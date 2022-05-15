from datetime import datetime
from typing import Dict, Optional

from django.core.management.base import BaseCommand, CommandParser
from pandas import DataFrame

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.exchanges.api import exchange_api
from cryptofeed_werks.lib import parse_period_from_to
from cryptofeed_werks.models import AggregatedTradeData, Symbol


class Command(BaseCommand):
    help = "Get trades from exchange API or S3"
    target = exchange_api

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("exchange", type=Exchange, choices=Exchange.values)
        parser.add_argument("symbol")
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> tuple:
        exchange = options["exchange"]
        symbol = options["symbol"]
        retry = options["retry"]
        verbose = bool(options["verbosity"])
        try:
            symbol = Symbol.objects.get(exchange=exchange, api_symbol=symbol)
        except Symbol.DoesNotExist:
            print(f"{exchange} {symbol} not registered.")
        else:
            timestamp_from, timestamp_to = parse_period_from_to(
                date_from=options["date_from"],
                time_from=options["time_from"],
                date_to=options["date_to"],
                time_to=options["time_to"],
            )

            def on_data_frame(
                symbol: Symbol,
                timestamp_from: datetime,
                timestamp_to: datetime,
                data_frame: DataFrame,
                validated: Optional[Dict[datetime, bool]] = {},
            ) -> DataFrame:
                """On data_frame, write aggregated data."""
                AggregatedTradeData.write(
                    symbol,
                    timestamp_from,
                    timestamp_to,
                    data_frame,
                    validated=validated,
                )

            exchange_api(
                symbol=symbol,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                on_data_frame=on_data_frame,
                retry=retry,
                verbose=verbose,
            )
