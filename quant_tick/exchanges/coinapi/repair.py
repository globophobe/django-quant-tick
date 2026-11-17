"""Repair native trade gaps with optional CoinAPI history.

Persist validated native minutes first, then replace whole missing minutes with
vendor data. Files precede REST and are checked against validation candles. Empty
vendor replies leave gaps retryable, and failures propagate after native progress
has been saved. Existing failed partitions follow the collector's retry policy.
All retained WebSocket rows are eligible. Capture them before archive or candle
requests to survive buffer cleanup, then refresh after fetching candles to pick
up newly arrived trades.
"""

import pandas as pd

from quant_tick.lib import (
    has_zero_trade_candle,
    iter_missing,
    iter_timeframe,
    iter_window,
)
from quant_tick.models import TradeData, WebSocketData

from .api import coinapi_history_start
from .controllers import CoinAPITrades


class CoinAPIRepairTrades(CoinAPITrades):
    write_unvalidated_websocket_partitions = False

    def get_native_candles(self, timestamp_from, timestamp_to):
        raise NotImplementedError

    def get_validation_candles(self, timestamp_from, timestamp_to, native):
        return native

    def on_rest_data_frame(self, timestamp_from, timestamp_to, candles):
        frame = self.fetch_trade_data_frame(timestamp_from, timestamp_to, candles)
        if frame.empty and not self.has_zero_trade_candles(
            candles, timestamp_from, timestamp_to
        ):
            return frame
        candles = self.validation_candles(frame, candles)
        self.on_data_frame(self.symbol, timestamp_from, timestamp_to, frame, candles)
        return frame

    def get_websocket_rows(self, timestamp_from, timestamp_to):
        return {
            row.timestamp: row
            for row in WebSocketData.objects.for_symbol(self.symbol).filter(
                timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
            )
        }

    def main(self, *, websocket_rows=None):
        if websocket_rows is None:
            websocket_rows = self.get_websocket_rows(
                self.timestamp_from, self.timestamp_to
            )
        for start, end in self.iter_missing_windows():
            candles = self.get_native_candles(start, end)
            websocket_rows.update(self.get_websocket_rows(start, end))
            covered = []
            for minute, next_minute in iter_window(start, end, value="1min"):
                row = websocket_rows.get(minute)
                frames = (
                    self.validate_websocket_partition(row, next_minute, candles)
                    if row is not None
                    else None
                )
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
                validation_candles = self.get_validation_candles(
                    fetch_from, fetch_to, native
                )
                for window_from, window_to in iter_timeframe(
                    fetch_from, fetch_to, value="1h", reverse=True
                ):
                    self.on_rest_data_frame(window_from, window_to, validation_candles)
