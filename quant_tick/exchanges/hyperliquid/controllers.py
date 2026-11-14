from collections.abc import Callable
from datetime import datetime
from functools import cached_property

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import Frequency, SymbolType, TradeDataRetry
from quant_tick.controllers import ExchangeWebSocket
from quant_tick.controllers.iterators import TradeDataIterator
from quant_tick.lib import has_zero_trade_candle, iter_missing, iter_window
from quant_tick.models import Symbol, TradeData

from ..coinapi.controllers import CoinAPITrades
from .candles import (
    fill_coinapi_candles,
    hyperliquid_candles,
    native_hyperliquid_candles,
)
from .sources import coinapi_history_start, coinapi_symbol


def hyperliquid_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Keep native trades where usable; backfill gaps with optional CoinAPI."""
    controller = HyperliquidTradesWebSocket
    if symbol.symbol_type == SymbolType.PERPETUAL and coinapi_symbol(symbol.api_symbol):
        controller = HyperliquidTrades
    controller(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class HyperliquidTradesWebSocket(ExchangeWebSocket):
    """Hyperliquid trades WebSocket."""

    def get_candles(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> DataFrame:
        return hyperliquid_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            resolution="1m",
            allow_coinapi=False,
        )

    def validate_websocket_partition(self, row, timestamp_to, candles):
        # Removing a vendor key must not downgrade previously validated history.
        if TradeData.objects.filter(
            symbol=self.symbol,
            timestamp=row.timestamp,
            frequency=Frequency.MINUTE,
            ok=True,
        ).exists():
            return None
        frames = super().validate_websocket_partition(row, timestamp_to, candles)
        if frames is not None:
            for frame in frames:
                if frame is not None:
                    frame["data_source"] = "hyperliquid"
        return frames


class HyperliquidTrades(CoinAPITrades, HyperliquidTradesWebSocket):
    """Use one source per complete minute, including on repair retries.

    Validated native minutes are persisted first. CoinAPI fills only uncovered
    intervals; no native and vendor trades are concatenated within a minute.
    An unavailable key retains the native-only collector. Provider failures
    leave missing ranges retryable instead of recording empty success.
    """

    write_unvalidated_websocket_partitions = False

    @cached_property
    def api_symbol(self):
        return coinapi_symbol(self.symbol.api_symbol)

    def on_rest_data_frame(self, timestamp_from, timestamp_to, candles):
        frame = self.fetch_trade_data_frame(timestamp_from, timestamp_to, candles)
        if frame.empty and not self.has_zero_trade_candles(
            candles, timestamp_from, timestamp_to
        ):
            return frame
        self.on_data_frame(self.symbol, timestamp_from, timestamp_to, frame, candles)
        return frame

    def main(self):
        for start, end in TradeDataIterator(self.symbol).iter_all(
            self.timestamp_from, self.timestamp_to, retry=self.retry
        ):
            candles = native_hyperliquid_candles(self.symbol.api_symbol, start, end)
            partitions = self.validate_websocket_partitions(start, end, candles)
            covered = []
            for minute, next_minute in iter_window(start, end, value="1min"):
                frames = partitions.get(minute)
                if frames is not None:
                    validation = TradeData._get_validation_frame(
                        **self.get_data_frame_kwargs(*frames)
                    )
                    self.on_data_frame(
                        self.symbol,
                        minute,
                        next_minute,
                        validation,
                        candles,
                        **self.get_data_frame_kwargs(*frames),
                    )
                elif has_zero_trade_candle(
                    self.symbol.exchange, candles, minute, next_minute
                ):
                    self.on_data_frame(
                        self.symbol, minute, next_minute, pd.DataFrame([]), candles
                    )
                else:
                    continue
                covered.append(minute)
            for fetch_from, fetch_to in iter_missing(start, end, covered, value="1min"):
                history_start = coinapi_history_start(self.api_symbol)
                if history_start is None or end <= history_start:
                    return
                fetch_from = max(fetch_from, history_start.ceil("1min"))
                if fetch_from >= fetch_to:
                    continue
                native = candles.loc[
                    (candles.index >= fetch_from) & (candles.index < fetch_to)
                ]
                validation_candles = fill_coinapi_candles(
                    self.symbol.api_symbol,
                    fetch_from,
                    fetch_to,
                    native,
                    resolution="1m",
                )
                self.on_rest_data_frame(fetch_from, fetch_to, validation_candles)
