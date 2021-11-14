from datetime import datetime, time, timezone
from typing import Callable

import pandas as pd

from cryptofeed_werks.controllers import ExchangeREST
from cryptofeed_werks.lib import get_current_time
from cryptofeed_werks.models import Symbol

from .base import BitflyerMixin


def bitflyer_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    verbose: bool = False,
):
    thirty_one_days_ago = datetime.combine(
        get_current_time().date() - pd.Timedelta("31d"), time.min
    )
    # UTC, please
    thirty_one_days_ago = thirty_one_days_ago.replace(tzinfo=timezone.utc)
    if timestamp_from.date() < thirty_one_days_ago:
        timestamp_from_display = timestamp_from.isoformat()
        thirty_one_days_ago_display = thirty_one_days_ago.isoformat()
        print(
            "Bitflyer limits trades to past 31 days, "
            f"{timestamp_from_display} modified to {thirty_one_days_ago_display}."
        )
        timestamp_from = thirty_one_days_ago
    BitflyerTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        verbose=verbose,
    ).main()


class BitflyerTrades(BitflyerMixin, ExchangeREST):
    pass
