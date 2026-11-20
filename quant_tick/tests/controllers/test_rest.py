import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import RETRY_INDETERMINATE, Exchange, FileData, Frequency
from quant_tick.controllers import (
    ExchangeREST,
    ExchangeWebSocket,
    increment_api_total_requests,
    is_terminal_page,
    throttle_api_requests,
)
from quant_tick.controllers.repair import ExchangeRepair
from quant_tick.models import TradeData, WebSocketData
from quant_tick.storage import convert_trade_data_to_daily, get_compact_max_timestamp_to

from ..base import BaseSymbolTest


class RateLimitTest(SimpleTestCase):
    def test_request_budget_initializes_and_resets_at_window_end(self):
        reset_key, total_key = "TEST_API_RESET", "TEST_API_TOTAL"
        with (
            patch.dict(os.environ),
            time_machine.travel(1000.0, tick=False) as clock,
            patch("quant_tick.controllers.rest.time.sleep") as sleep,
        ):
            os.environ.pop(reset_key, None)
            os.environ.pop(total_key, None)
            throttle_api_requests(reset_key, total_key, 60, 10)
            self.assertEqual(float(os.environ[reset_key]), 1060)
            self.assertEqual(os.environ[total_key], "0")

            increment_api_total_requests(total_key)
            clock.shift(59)
            throttle_api_requests(reset_key, total_key, 60, 10)
            self.assertEqual(os.environ[total_key], "1")
            self.assertEqual(float(os.environ[reset_key]), 1060)

            clock.shift(1)
            throttle_api_requests(reset_key, total_key, 60, 10)
            self.assertEqual(os.environ[total_key], "0")
            self.assertEqual(float(os.environ[reset_key]), 1120)
            sleep.assert_not_called()

            for _ in range(10):
                increment_api_total_requests(total_key)
            clock.shift(59)
            sleep.side_effect = clock.shift
            throttle_api_requests(reset_key, total_key, 60, 10)
            sleep.assert_called_once_with(1)
            self.assertEqual(os.environ[total_key], "0")
            self.assertEqual(float(os.environ[reset_key]), 1180)
            increment_api_total_requests(total_key)
            throttle_api_requests(reset_key, total_key, 60, 10)
            self.assertEqual(os.environ[total_key], "1")
            sleep.assert_called_once_with(1)


class FixedIntervalTerminalPageTest(SimpleTestCase):
    def test_short_page_spanning_complete_time_window_is_not_terminal(self):
        timestamp_from = datetime(2025, 10, 22, 12, tzinfo=UTC)
        data = [
            timestamp_from + timedelta(hours=hour)
            for hour in range(300)
            if hour not in {12, 53, 109, 208, 240}
        ]

        self.assertFalse(
            is_terminal_page(
                data,
                get_timestamp=lambda item: item,
                interval=timedelta(hours=1),
                max_results=300,
            )
        )

    def test_short_page_with_short_time_span_is_terminal(self):
        timestamp_from = datetime(2025, 10, 22, 12, tzinfo=UTC)
        data = [timestamp_from + timedelta(hours=hour) for hour in range(295)]

        self.assertTrue(
            is_terminal_page(
                data,
                get_timestamp=lambda item: item,
                interval=timedelta(hours=1),
                max_results=300,
            )
        )


class DummyExchangeREST(ExchangeREST):
    def __init__(
        self,
        *args,
        api_results: list[tuple[list[dict], bool, str | None]],
        **kwargs,
    ) -> None:
        self.frames = []
        self.candle_calls = []
        self.api_results = api_results
        self.api_calls = 0
        self.pagination_ids = []
        super().__init__(*args, on_data_frame=self.capture_frame, **kwargs)

    def capture_frame(
        self,
        symbol,
        timestamp_from,
        timestamp_to,
        data_frame,
        candles,
        **kwargs,
    ):
        self.frames.append(
            (
                timestamp_from,
                timestamp_to,
                data_frame.copy(),
                candles,
                kwargs,
            )
        )

    def get_pagination_id(self, timestamp_to: datetime) -> str:
        return timestamp_to.isoformat()

    def iter_api(self, timestamp_from: datetime, pagination_id: str) -> tuple:
        self.pagination_ids.append(pagination_id)
        result = self.api_results[self.api_calls]
        self.api_calls += 1
        return result

    def parse_data(self, data: list) -> list:
        return data

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> pd.DataFrame:
        self.candle_calls.append((timestamp_from, timestamp_to))
        return pd.DataFrame([])


class DummyPartitionREST(DummyExchangeREST):
    partition_scoped = True


class DummyExchangeWebSocket(DummyExchangeREST, ExchangeWebSocket):
    pass


class DummyRepairTrades(DummyExchangeREST, ExchangeRepair):
    columns = ExchangeREST.columns
    on_rest_data_frame = ExchangeREST.on_rest_data_frame

    def get_native_candles(self, timestamp_from, timestamp_to):
        return self.get_candles(timestamp_from, timestamp_to)

    def iter_missing_windows(self):
        yield self.timestamp_from, self.timestamp_to


COLLECTION_MODES = (
    DummyExchangeREST,
    DummyPartitionREST,
    DummyExchangeWebSocket,
    DummyRepairTrades,
)


@time_machine.travel(datetime(2009, 1, 3, tzinfo=UTC), tick=False)
class ExchangeRESTTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.one_minute = pd.Timedelta("1min")
        self.symbol = self.get_symbol()

    def get_trade(self, uid: int, minute: int) -> dict:
        return {
            "uid": str(uid),
            "timestamp": self.timestamp_from + (self.one_minute * minute),
            "nanoseconds": 0,
            "price": Decimal(100),
            "volume": Decimal(100),
            "notional": Decimal(1),
            "tickRule": 1,
            "index": uid,
        }

    def create_websocket_data(self, uid: int, minute: int) -> WebSocketData:
        return WebSocketData.objects.create(
            exchange=self.symbol.exchange,
            api_symbol=self.symbol.api_symbol,
            significant_trade_filter=self.symbol.significant_trade_filter or 0,
            timestamp=self.timestamp_from + (self.one_minute * minute),
            raw_trades=[self.get_trade(uid, minute)],
        )

    def get_candles(self, values_by_minute: dict[int, Decimal]) -> pd.DataFrame:
        return pd.DataFrame(
            {"notional": list(values_by_minute.values())},
            index=[
                self.timestamp_from + (self.one_minute * minute)
                for minute in values_by_minute
            ],
        )

    def test_main_reuses_next_partition_buffer(self):
        ts0 = self.timestamp_from
        partitions = [
            (ts0 + (self.one_minute * 5), ts0 + (self.one_minute * 10)),
            (ts0, ts0 + (self.one_minute * 5)),
        ]
        api_results = [
            (
                [
                    self.get_trade(4, 9),
                    self.get_trade(3, 8),
                    self.get_trade(2, 4),
                    self.get_trade(1, 3),
                ],
                True,
                None,
            )
        ]
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 10),
            retry=False,
            verbose=False,
            api_results=api_results,
        )

        with patch(
            "quant_tick.controllers.rest.TradeDataIterator.iter_all",
            return_value=partitions,
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(len(controller.frames), 2)
        first = controller.frames[0][2]
        second = controller.frames[1][2]
        self.assertEqual(
            list(first.timestamp),
            [ts0 + (self.one_minute * 8), ts0 + (self.one_minute * 9)],
        )
        self.assertEqual(
            list(second.timestamp),
            [ts0 + (self.one_minute * 3), ts0 + (self.one_minute * 4)],
        )

    def test_main_fetches_candles_for_iterator_window(self):
        ts0 = self.timestamp_from
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + self.one_minute,
            retry=False,
            verbose=False,
            api_results=[([self.get_trade(0, 0)], True, None)],
        )

        with patch(
            "quant_tick.controllers.rest.TradeDataIterator.iter_all",
            return_value=[(ts0, ts0 + self.one_minute)],
        ):
            controller.main()

        self.assertEqual(controller.candle_calls, [(ts0, ts0 + self.one_minute)])

    def test_main_splices_valid_websocket_partition_with_rest_backfill(self):
        ts0 = self.timestamp_from
        self.create_websocket_data(1, 1)
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0 + self.one_minute,
            timestamp_to=ts0 + (self.one_minute * 3),
            retry=False,
            verbose=False,
            api_results=[([self.get_trade(2, 2)], True, None)],
        )

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0 + self.one_minute, ts0 + (self.one_minute * 3))],
            ),
            patch(
                "quant_tick.controllers.rest.TradeData.validate",
                return_value=True,
            ) as mock_validate,
        ):
            controller.main()

        self.assertEqual(mock_validate.call_count, 1)
        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(
            [(frame[0], frame[1]) for frame in controller.frames],
            [(ts0 + self.one_minute, ts0 + (self.one_minute * 3))],
        )
        self.assertEqual(list(controller.frames[0][2].uid), ["1", "2"])
        self.assertIn("raw_trades", controller.frames[0][4])

    def test_main_accepts_missing_websocket_minutes_when_candles_have_no_trades(self):
        ts0 = self.timestamp_from
        for minute in (0, 1, 3, 5):
            self.create_websocket_data(1000 + minute, minute)

        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 6),
            retry=False,
            verbose=False,
            api_results=[],
        )
        candles = self.get_candles({2: Decimal(0), 4: Decimal(0)})

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 6))],
            ),
            patch.object(controller, "get_candles", return_value=candles),
            patch("quant_tick.controllers.rest.TradeData.validate", return_value=True),
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 0)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + (self.one_minute * 6))
        self.assertEqual(list(frame.uid), ["1000", "1001", "1003", "1005"])
        self.assertIn("raw_trades", kwargs)

    def test_main_accepts_empty_websocket_window_when_candles_have_no_trades(self):
        ts0 = self.timestamp_from
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 3),
            retry=False,
            verbose=False,
            api_results=[],
        )
        candles = self.get_candles(
            {
                0: Decimal(0),
                1: Decimal(0),
                2: Decimal(0),
            }
        )

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 3))],
            ),
            patch.object(controller, "get_candles", return_value=candles),
            patch("quant_tick.controllers.rest.TradeData.validate") as mock_validate,
        ):
            controller.main()

        mock_validate.assert_not_called()
        self.assertEqual(controller.api_calls, 0)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + (self.one_minute * 3))
        self.assertEqual(len(frame), 0)
        self.assertEqual(kwargs, {})

    def test_main_accepts_missing_bitfinex_websocket_minutes_when_candles_are_omitted(
        self,
    ):
        ts0 = self.timestamp_from
        self.symbol = self.get_symbol(
            exchange=Exchange.BITFINEX,
            api_symbol="tBTCF0:USTF0",
        )
        for minute in (0, 2):
            self.create_websocket_data(1000 + minute, minute)

        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 4),
            retry=False,
            verbose=False,
            api_results=[],
        )
        candles = self.get_candles({0: Decimal(1), 2: Decimal(1)})

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 4))],
            ),
            patch.object(controller, "get_candles", return_value=candles),
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 0)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + (self.one_minute * 4))
        self.assertEqual(list(frame.uid), ["1000", "1002"])
        self.assertIn("raw_trades", kwargs)

    def test_main_accepts_empty_bitfinex_websocket_window_when_candles_are_omitted(
        self,
    ):
        ts0 = self.timestamp_from
        self.symbol = self.get_symbol(
            exchange=Exchange.BITFINEX,
            api_symbol="tBTCF0:USTF0",
        )
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 3),
            retry=False,
            verbose=False,
            api_results=[],
        )

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 3))],
            ),
            patch.object(controller, "get_candles", return_value=pd.DataFrame([])),
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 0)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + (self.one_minute * 3))
        self.assertEqual(len(frame), 0)
        self.assertEqual(kwargs, {})

    def test_main_backfills_only_invalid_websocket_minutes_in_hour_partition(self):
        ts0 = self.timestamp_from
        invalid_minute = 42
        for minute in range(60):
            self.create_websocket_data(1000 + minute, minute)

        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + pd.Timedelta("1h"),
            retry=False,
            verbose=False,
            api_results=[
                ([self.get_trade(9000 + invalid_minute, invalid_minute)], True, None)
            ],
        )

        def validate_partition(
            symbol,
            timestamp_from,
            timestamp_to,
            candles,
            *,
            raw_trades=None,
            aggregated_trades=None,
            filtered_trades=None,
        ):
            return raw_trades.iloc[0].uid != str(1000 + invalid_minute)

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + pd.Timedelta("1h"))],
            ),
            patch(
                "quant_tick.controllers.rest.TradeData.validate",
                side_effect=validate_partition,
            ) as mock_validate,
        ):
            controller.main()

        self.assertEqual(mock_validate.call_count, 60)
        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + pd.Timedelta("1h"))
        self.assertEqual(len(frame), 60)
        self.assertEqual(frame.iloc[0].uid, "1000")
        self.assertEqual(frame.iloc[invalid_minute].uid, str(9000 + invalid_minute))
        self.assertEqual(
            frame.iloc[invalid_minute - 1].uid,
            str(1000 + invalid_minute - 1),
        )
        self.assertEqual(
            frame.iloc[invalid_minute + 1].uid,
            str(1000 + invalid_minute + 1),
        )
        self.assertIn("raw_trades", kwargs)
        self.assertEqual(list(kwargs["raw_trades"].uid), list(frame.uid))

    def test_main_replaces_span_between_two_invalid_websocket_gaps_with_rest(self):
        ts0 = self.timestamp_from
        invalid_minutes = {2, 4}
        for minute in range(6):
            self.create_websocket_data(1000 + minute, minute)

        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 6),
            retry=False,
            verbose=False,
            api_results=[
                (
                    [
                        self.get_trade(9000 + minute, minute)
                        for minute in range(5, -1, -1)
                    ],
                    True,
                    None,
                )
            ],
        )

        def validate_partition(
            symbol,
            timestamp_from,
            timestamp_to,
            candles,
            *,
            raw_trades=None,
            aggregated_trades=None,
            filtered_trades=None,
        ):
            uid = int(raw_trades.iloc[0].uid)
            minute = uid - 1000
            return minute not in invalid_minutes

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 6))],
            ),
            patch(
                "quant_tick.controllers.rest.TradeData.validate",
                side_effect=validate_partition,
            ),
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + (self.one_minute * 6))
        self.assertEqual(
            list(frame.uid),
            [
                str(9000 + minute) if 2 <= minute <= 4 else str(1000 + minute)
                for minute in range(6)
            ],
        )
        self.assertIn("raw_trades", kwargs)

    def test_main_replaces_sparse_websocket_gap_span_when_ranges_are_within_limit(self):
        ts0 = self.timestamp_from
        invalid_minutes = {2, 20}
        for minute in range(30):
            self.create_websocket_data(1000 + minute, minute)

        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 30),
            retry=False,
            verbose=False,
            api_results=[
                (
                    [
                        self.get_trade(9000 + minute, minute)
                        for minute in range(29, -1, -1)
                    ],
                    True,
                    None,
                )
            ],
        )

        def validate_partition(
            symbol,
            timestamp_from,
            timestamp_to,
            candles,
            *,
            raw_trades=None,
            aggregated_trades=None,
            filtered_trades=None,
        ):
            uid = int(raw_trades.iloc[0].uid)
            minute = uid - 1000
            return minute not in invalid_minutes

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 30))],
            ),
            patch(
                "quant_tick.controllers.rest.TradeData.validate",
                side_effect=validate_partition,
            ),
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + (self.one_minute * 30))
        self.assertEqual(
            list(frame.uid),
            [
                str(9000 + minute) if 2 <= minute <= 20 else str(1000 + minute)
                for minute in range(30)
            ],
        )
        self.assertIn("raw_trades", kwargs)

    def test_main_uses_rest_only_when_invalid_websocket_gaps_exceed_max_ranges(self):
        ts0 = self.timestamp_from
        invalid_minutes = {1, 3, 5}
        for minute in range(10):
            self.create_websocket_data(1000 + minute, minute)

        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 10),
            retry=False,
            verbose=False,
            api_results=[
                (
                    [
                        self.get_trade(9000 + minute, minute)
                        for minute in range(9, -1, -1)
                    ],
                    True,
                    None,
                )
            ],
        )

        def validate_partition(
            symbol,
            timestamp_from,
            timestamp_to,
            candles,
            *,
            raw_trades=None,
            aggregated_trades=None,
            filtered_trades=None,
        ):
            uid = int(raw_trades.iloc[0].uid)
            minute = uid - 1000
            return minute not in invalid_minutes

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 10))],
            ),
            patch(
                "quant_tick.controllers.rest.TradeData.validate",
                side_effect=validate_partition,
            ),
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        timestamp_from, timestamp_to, frame, _candles, kwargs = controller.frames[0]
        self.assertEqual(timestamp_from, ts0)
        self.assertEqual(timestamp_to, ts0 + (self.one_minute * 10))
        self.assertEqual(list(frame.uid), [str(9000 + minute) for minute in range(10)])
        self.assertEqual(kwargs, {})

    def test_retry_preserves_each_collection_modes_source_policy(self):
        start, end = self.timestamp_from, self.timestamp_from + self.one_minute
        self.create_websocket_data(1, 0)
        candles = self.get_candles({0: Decimal(1)})
        for mode in COLLECTION_MODES:
            for retry in (True, RETRY_INDETERMINATE):
                with self.subTest(mode=mode.__name__, retry=retry):
                    controller = mode(
                        self.symbol,
                        start,
                        end,
                        retry=retry,
                        verbose=False,
                        api_results=[([self.get_trade(9001, 0)], True, None)],
                    )
                    with (
                        patch(
                            "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                            return_value=[(start, end)],
                        ),
                        patch.object(
                            controller,
                            "get_websocket_rows",
                            wraps=controller.get_websocket_rows,
                        ) as rows,
                        patch.object(controller, "get_candles", return_value=candles),
                        patch(
                            "quant_tick.controllers.repair.ExchangeRepair.history_start"
                        ) as history,
                    ):
                        controller.main()
                    history.assert_not_called()
                    if mode is not DummyRepairTrades:
                        rows.assert_not_called()
                    uses_rest = mode in (DummyExchangeREST, DummyPartitionREST)
                    self.assertEqual(controller.api_calls, int(uses_rest))
                    if mode is DummyExchangeWebSocket:
                        self.assertEqual(controller.frames, [])
                    else:
                        self.assertEqual(len(controller.frames), 1)
                        self.assertEqual(
                            controller.frames[0][2].uid.tolist(),
                            ["9001" if uses_rest else "1"],
                        )

    def test_main_fetches_rest_when_websocket_partition_is_not_valid(self):
        ts0 = self.timestamp_from
        self.create_websocket_data(1, 1)
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0 + self.one_minute,
            timestamp_to=ts0 + (self.one_minute * 2),
            retry=False,
            verbose=False,
            api_results=[([self.get_trade(1, 1)], True, None)],
        )

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0 + self.one_minute, ts0 + (self.one_minute * 2))],
            ),
            patch(
                "quant_tick.controllers.rest.TradeData.validate",
                return_value=False,
            ),
        ):
            controller.main()

        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        self.assertEqual(controller.frames[0][0], ts0 + self.one_minute)
        self.assertEqual(controller.frames[0][1], ts0 + (self.one_minute * 2))
        self.assertEqual(controller.frames[0][4], {})

    def test_retained_websocket_survives_cleanup_and_refreshes_across_modes(self):
        start, end = self.timestamp_from, self.timestamp_from + self.one_minute
        for mode in COLLECTION_MODES:
            for scenario in (
                "cleanup",
                "archive_cleanup",
                "late_arrival",
                "completion",
            ):
                with self.subTest(mode=mode.__name__, scenario=scenario):
                    WebSocketData.objects.all().delete()
                    if scenario != "late_arrival":
                        row = self.create_websocket_data(1, 0)
                        if scenario == "completion":
                            row.raw_trades[0]["notional"] = Decimal("0.5")
                            row.raw_trades[0]["volume"] = Decimal(50)
                            row.save()
                    # A snapshot can contain rows outside the current window.
                    self.create_websocket_data(100, -1)
                    self.create_websocket_data(101, 1)
                    controller = mode(
                        self.symbol, start, end, api_results=[], verbose=False
                    )
                    candles = self.get_candles({0: Decimal(1)})
                    with time_machine.travel(
                        start + pd.Timedelta("59min"), tick=False
                    ) as clock:
                        kwargs = {}
                        if scenario == "archive_cleanup":
                            kwargs["websocket_rows"] = controller.get_websocket_rows(
                                start - self.one_minute, end + self.one_minute
                            )
                            WebSocketData.objects.all().delete()
                            clock.shift(timedelta(hours=2))

                        def fetch_candles(*args, scenario=scenario, candles=candles):
                            clock.shift(timedelta(minutes=3))
                            WebSocketData.objects.all().delete()
                            if scenario in ("late_arrival", "completion"):
                                self.create_websocket_data(2, 0)
                            return candles

                        with (
                            patch.object(
                                controller, "get_candles", side_effect=fetch_candles
                            ),
                            patch(
                                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                                return_value=[(start, end)],
                            ),
                            patch(
                                "quant_tick.controllers.repair.ExchangeRepair.history_start"
                            ) as history,
                        ):
                            controller.main(**kwargs)
                        history.assert_not_called()
                    self.assertEqual(controller.api_calls, 0)
                    self.assertEqual(len(controller.frames), 1)
                    frame = controller.frames[0][2]
                    self.assertEqual(
                        frame.uid.tolist(),
                        ["2" if scenario in ("late_arrival", "completion") else "1"],
                    )
                    self.assertEqual(frame.notional.sum(), Decimal(1))
                    self.assertIs(
                        TradeData.validate(
                            self.symbol, start, end, candles, raw_trades=frame
                        ),
                        True,
                    )

    def test_websocket_snapshot_only_loads_unvalidated_enabled_payloads(self):
        self.symbol.save_raw = False
        self.symbol.significant_trade_filter = 1000
        self.symbol.save()
        other_symbol = self.get_symbol(api_symbol="other")
        start = self.timestamp_from
        for frequency in (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY):
            for ok in (True, False, None):
                with self.subTest(frequency=frequency, ok=ok):
                    TradeData.objects.all().delete()
                    WebSocketData.objects.all().delete()
                    minutes = {0, frequency - 1, frequency, frequency + 1}
                    for minute in minutes:
                        trade = self.get_trade(minute, minute)
                        WebSocketData.objects.create(
                            exchange=self.symbol.exchange,
                            api_symbol=self.symbol.api_symbol,
                            significant_trade_filter=1000,
                            timestamp=start + minute * self.one_minute,
                            raw_trades=[trade],
                            aggregated_trades=[trade],
                            filtered_trades=[trade],
                        )
                    TradeData.objects.create(
                        symbol=self.symbol, timestamp=start, frequency=frequency, ok=ok
                    )
                    TradeData.objects.create(
                        symbol=self.symbol,
                        timestamp=start + frequency * self.one_minute,
                        frequency=Frequency.MINUTE,
                        ok=True,
                    )
                    TradeData.objects.create(
                        symbol=other_symbol,
                        timestamp=start + (frequency + 1) * self.one_minute,
                        frequency=Frequency.MINUTE,
                        ok=True,
                    )
                    # Hour/day coverage starts before the requested window.
                    first = 1 if frequency > 1 else 0
                    window_from = start + first * self.one_minute
                    window_to = start + (frequency + 2) * self.one_minute
                    controller = DummyExchangeREST(
                        self.symbol, window_from, window_to, api_results=[]
                    )
                    with self.assertNumQueries(1):
                        rows = controller.get_websocket_rows(window_from, window_to)
                    expected = {
                        start + minute * self.one_minute
                        for minute in minutes
                        if minute >= first
                        and minute != frequency
                        and (ok is not True or minute >= frequency)
                    }
                    self.assertEqual(set(rows), expected)
                    for row in rows.values():
                        self.assertTrue(
                            {"raw_trades", "aggregated_trades"}
                            <= row.get_deferred_fields()
                        )
                        with self.assertNumQueries(0):
                            raw, aggregated, filtered = row.get_data_frames(self.symbol)
                        self.assertIsNone(raw)
                        self.assertIsNone(aggregated)
                        self.assertEqual(len(filtered), 1)

    def test_websocket_main_skips_validated_minutes_and_compacted_history(self):
        start = self.timestamp_from
        for minute in (59, 60, 61, 62, 64):
            self.create_websocket_data(minute, minute)
        TradeData.objects.create(
            symbol=self.symbol, timestamp=start, frequency=Frequency.HOUR, ok=False
        )
        for minute, ok in ((60, True), (61, False), (62, None), (64, True)):
            TradeData.objects.create(
                symbol=self.symbol,
                timestamp=start + minute * self.one_minute,
                frequency=Frequency.MINUTE,
                ok=ok,
            )
        controller = DummyExchangeWebSocket(
            self.symbol,
            start + 59 * self.one_minute,
            start + 65 * self.one_minute,
            api_results=[],
        )
        candles = self.get_candles(
            {
                minute: Decimal(0 if minute in (63, 64) else 1)
                for minute in range(59, 65)
            }
        )
        with patch.object(controller, "get_candles", return_value=candles) as fetch:
            controller.main()
        fetch.assert_called_once_with(
            start + 61 * self.one_minute, start + 64 * self.one_minute
        )
        self.assertEqual(
            [frame[0] for frame in controller.frames],
            [start + minute * self.one_minute for minute in (63, 62, 61)],
        )
        self.assertTrue(controller.frames[0][2].empty)
        self.assertEqual(controller.frames[1][2].uid.tolist(), ["62"])
        self.assertEqual(controller.frames[2][2].uid.tolist(), ["61"])

        TradeData.objects.filter(frequency=Frequency.MINUTE).update(ok=True)
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=start + 63 * self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        controller.frames.clear()
        with patch.object(controller, "get_candles") as fetch:
            controller.main()
        fetch.assert_not_called()
        self.assertEqual(controller.frames, [])

    @override_settings(
        STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
    )
    def test_websocket_rechecks_compacted_coverage_after_candle_fetch(self):
        start = self.timestamp_from
        candles = self.get_candles({minute: Decimal(1) for minute in range(60)})
        for minute in range(60):
            timestamp = start + minute * self.one_minute
            TradeData.write(
                self.symbol,
                timestamp,
                timestamp + self.one_minute,
                candles,
                raw_trades=pd.DataFrame([self.get_trade(minute, minute)]),
            )
        TradeData.objects.filter(symbol=self.symbol).update(ok=None)
        self.create_websocket_data(10, 10)
        lower, upper = start + 10 * self.one_minute, start + 11 * self.one_minute
        controller = DummyExchangeWebSocket(self.symbol, lower, upper, api_results=[])
        retained = controller.get_websocket_rows(lower, upper)
        WebSocketData.objects.all().delete()

        def persist(symbol, lower, upper, frame, candles, **kwargs):
            return TradeData.write(symbol, lower, upper, candles, **kwargs)

        controller.on_data_frame = persist
        now = start + timedelta(hours=3)

        def fetch_candles(*args):
            convert_trade_data_to_daily(
                self.symbol, start, get_compact_max_timestamp_to(now)
            )
            return candles

        with (
            time_machine.travel(now, tick=False),
            patch.object(controller, "get_candles", side_effect=fetch_candles),
        ):
            controller.main(websocket_rows=retained)
        stored = TradeData.objects.get(symbol=self.symbol)
        self.assertEqual(stored.frequency, Frequency.HOUR)
        self.assertEqual(len(stored.get_data_frame(FileData.RAW)), 60)

    def test_websocket_validation_and_fallback_across_modes(self):
        start, end = self.timestamp_from, self.timestamp_from + self.one_minute
        for mode in COLLECTION_MODES:
            for scenario, quantity, expected_candle in (
                ("valid", "1", "1"),
                ("invalid", "0.5", "1"),
                ("unknown", "1", None),
                ("missing", None, "1"),
                ("quiet", None, "0"),
            ):
                with self.subTest(mode=mode.__name__, scenario=scenario):
                    WebSocketData.objects.all().delete()
                    if quantity is not None:
                        row = self.create_websocket_data(1, 0)
                        row.raw_trades[0]["notional"] = Decimal(quantity)
                        row.raw_trades[0]["volume"] = Decimal(quantity) * 100
                        row.save()
                    candles = self.get_candles(
                        {0: Decimal(expected_candle)}
                        if expected_candle is not None
                        else {}
                    )
                    controller = mode(
                        self.symbol,
                        start,
                        end,
                        verbose=False,
                        api_results=[([self.get_trade(9001, 0)], True, None)],
                    )
                    with (
                        time_machine.travel(
                            start + pd.Timedelta("59min"), tick=False
                        ) as clock,
                        patch(
                            "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                            return_value=[(start, end)],
                        ),
                        patch(
                            "quant_tick.controllers.repair.ExchangeRepair.history_start",
                            return_value=pd.Timestamp(start),
                        ),
                    ):

                        def fetch_candles(*args, candles=candles):
                            clock.shift(timedelta(minutes=3))
                            return candles

                        with patch.object(
                            controller, "get_candles", side_effect=fetch_candles
                        ):
                            controller.main()
                    uses_rest = mode is not DummyExchangeWebSocket and scenario in (
                        "invalid",
                        "unknown",
                        "missing",
                    )
                    self.assertEqual(controller.api_calls, int(uses_rest))
                    if mode is DummyExchangeWebSocket and scenario == "missing":
                        self.assertEqual(controller.frames, [])
                        continue
                    self.assertEqual(len(controller.frames), 1)
                    frame = controller.frames[0][2]
                    expected_ids = (
                        [] if scenario == "quiet" else ["9001" if uses_rest else "1"]
                    )
                    self.assertEqual(
                        frame.uid.tolist() if not frame.empty else [], expected_ids
                    )
                    validation = TradeData.validate(
                        self.symbol, start, end, candles, raw_trades=frame
                    )
                    expected_ok = (
                        None
                        if scenario == "unknown"
                        else not (scenario == "invalid" and not uses_rest)
                    )
                    self.assertIs(validation, expected_ok)

    def test_main_resets_pagination_across_partition_gaps(self):
        ts0 = self.timestamp_from
        partitions = [
            (ts0 + (self.one_minute * 9), ts0 + (self.one_minute * 10)),
            (ts0, ts0 + self.one_minute),
        ]
        api_results = [
            (
                [
                    self.get_trade(9, 9),
                    self.get_trade(8, 8),
                ],
                False,
                "stale-cursor",
            ),
            ([self.get_trade(0, 0)], True, None),
        ]
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 10),
            retry=False,
            verbose=False,
            api_results=api_results,
        )

        with patch(
            "quant_tick.controllers.rest.TradeDataIterator.iter_all",
            return_value=partitions,
        ):
            controller.main()

        self.assertEqual(
            controller.pagination_ids,
            [
                (ts0 + (self.one_minute * 10)).isoformat(),
                (ts0 + self.one_minute).isoformat(),
            ],
        )
        self.assertEqual(controller.api_calls, 2)
        self.assertEqual(len(controller.frames), 2)
