import logging
from collections.abc import Callable
from datetime import date, datetime

from pandas import DataFrame

from quant_tick.controllers import ExchangeREST
from quant_tick.models import Symbol

from .base import CoinbaseMixin
from .constants import BTCUSD

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None

logger = logging.getLogger(__name__)


def coinbase_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
) -> None:
    """Get Coinbase trades."""
    CoinbaseTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class CoinbaseTrades(CoinbaseMixin, ExchangeREST):
    """Coinbase trades."""

    def assert_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        trades: list[dict] | None = None,
    ) -> None:
        """Assert data frame."""
        super().assert_data_frame(timestamp_from, timestamp_to, data_frame, trades)
        # Missing orders.
        expected = len(trades) - 1
        if self.symbol.api_symbol == BTCUSD:
            # It seems 45 ids may have been skipped for BTC-USD on 2021-06-09
            if timestamp_from.date() == date(2021, 6, 9):
                return
            # There was a missing order for BTC-USD on 2019-04-11
            elif timestamp_from.date() == date(2019, 4, 11):
                return
        diff = data_frame["index"].diff().dropna()
        actual = abs(diff.sum())
        if actual != expected:
            message = (
                "Coinbase trade sequence gap: "
                f"{self.symbol.api_symbol} {timestamp_from} {timestamp_to} "
                f"expected={expected} actual={actual}"
            )
            logger.warning(message)
            if sentry_sdk:
                sentry_sdk.capture_message(message, level="error")
