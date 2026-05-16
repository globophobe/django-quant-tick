from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import Frequency
from quant_tick.controllers import ExchangeREST, ExchangeS3, TradeDataIterator
from quant_tick.exchanges.bitmex.base import BitmexS3Mixin
from quant_tick.models import TradeData, WebSocketData

from ..base import BaseSymbolTest


@time_machine.travel(datetime(2009, 1, 3), tick=False)
class TradeDataIteratorTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.one_minute = pd.Timedelta("1min")
        self.timestamp_to = self.timestamp_from + (self.one_minute * 5)
        self.symbol = self.get_symbol()

    def get_values(self, retry: bool = False) -> list[tuple[datetime, datetime]]:
        return [
            value
            for value in TradeDataIterator(self.symbol).iter_all(
                timestamp_from=self.timestamp_from,
                timestamp_to=self.timestamp_to,
                retry=retry,
            )
        ]

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 3).replace(tzinfo=UTC),
    )
    def test_iter_all_with_no_results(self, mock_get_max_timestamp_to):
        values = self.get_values()
        self.assertEqual(len(values), 0)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_iter_all_with_head(self, mock_get_max_timestamp_to):
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_iter_all_with_one_ok(self, mock_get_max_timestamp_to):
        obj = TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0], obj.timestamp + self.one_minute)
        self.assertEqual(values[0][1], self.timestamp_to)
        self.assertEqual(values[-1][0], self.timestamp_from)
        self.assertEqual(values[-1][1], obj.timestamp)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_iter_all_with_two_ok(self, mock_get_max_timestamp_to):
        obj_one = TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        obj_two = TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + (self.one_minute * 3),
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0][0], self.timestamp_to - self.one_minute)
        self.assertEqual(values[0][1], self.timestamp_to)
        self.assertEqual(values[1][0], obj_one.timestamp + self.one_minute)
        self.assertEqual(values[1][1], obj_two.timestamp)
        self.assertEqual(values[-1][0], self.timestamp_from)
        self.assertEqual(values[-1][1], obj_one.timestamp)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_iter_all_with_tail(self, mock_get_max_timestamp_to):
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_to - self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_to - self.one_minute)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_iter_all_with_retry_and_one_not_ok(self, mock_get_max_timestamp_to):
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=False,
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_iter_all_with_retry_and_one_missing(self, mock_get_max_timestamp_to):
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=None,
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_iter_all_ignores_overlapping_coverage_when_minutes_are_complete(
        self, mock_get_max_timestamp_to
    ):
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.DAY,
            ok=True,
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(values, [])


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


class DummyExchangeS3(ExchangeS3):
    def __init__(self, *args, data_frame: pd.DataFrame, **kwargs) -> None:
        self.frames = []
        self.candle_calls = []
        self.download_calls = 0
        self.data_frame = data_frame
        super().__init__(*args, on_data_frame=self.capture_frame, **kwargs)

    def capture_frame(self, symbol, timestamp_from, timestamp_to, data_frame, candles):
        self.frames.append((timestamp_from, timestamp_to, data_frame.copy()))

    def get_url(self, date: datetime.date) -> str:
        return f"https://example.test/{date.isoformat()}.csv.gz"

    def get_data_frame(self, date: datetime.date) -> pd.DataFrame | None:
        self.download_calls += 1
        return self.data_frame.copy()

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> pd.DataFrame:
        self.candle_calls.append((timestamp_from, timestamp_to))
        return pd.DataFrame([])


class DummyBitmexExchangeS3(BitmexS3Mixin, DummyExchangeS3):
    def get_data_frame(self, date: datetime.date) -> pd.DataFrame | None:
        return DummyExchangeS3.get_data_frame(self, date)

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> pd.DataFrame:
        return DummyExchangeS3.get_candles(self, timestamp_from, timestamp_to)


class DummyMissingBitmexExchangeS3(DummyBitmexExchangeS3):
    missing_archive_dates = frozenset({datetime(2009, 1, 3).date()})

    def get_data_frame(self, date: datetime.date) -> pd.DataFrame | None:
        self.download_calls += 1
        return None


@time_machine.travel(datetime(2009, 1, 3), tick=False)
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
            "price": Decimal("100"),
            "volume": Decimal("100"),
            "notional": Decimal("1"),
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
        candles = self.get_candles({2: Decimal("0"), 4: Decimal("0")})

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
                0: Decimal("0"),
                1: Decimal("0"),
                2: Decimal("0"),
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

    def test_main_uses_rest_only_for_two_invalid_gaps_in_short_partition(self):
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
        self.assertEqual(list(frame.uid), [str(9000 + minute) for minute in range(6)])
        self.assertEqual(kwargs, {})

    def test_main_splices_sparse_websocket_gaps_when_ranges_are_within_limit(self):
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
                str(9000 + minute) if minute in invalid_minutes else str(1000 + minute)
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

    def test_main_uses_rest_only_on_retry_even_when_websocket_partition_is_valid(self):
        ts0 = self.timestamp_from
        self.create_websocket_data(1, 1)
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0 + self.one_minute,
            timestamp_to=ts0 + (self.one_minute * 2),
            retry=True,
            verbose=False,
            api_results=[([self.get_trade(9001, 1)], True, None)],
        )

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0 + self.one_minute, ts0 + (self.one_minute * 2))],
            ),
            patch(
                "quant_tick.controllers.rest.WebSocketData.objects.for_symbol",
            ) as mock_for_symbol,
        ):
            controller.main()

        mock_for_symbol.assert_not_called()
        self.assertEqual(controller.api_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        self.assertEqual(list(controller.frames[0][2].uid), ["9001"])
        self.assertEqual(controller.frames[0][4], {})

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

    def test_main_skips_websocket_lookup_before_retention_window(self):
        ts0 = self.timestamp_from
        controller = DummyExchangeREST(
            self.symbol,
            timestamp_from=ts0,
            timestamp_to=ts0 + (self.one_minute * 2),
            retry=False,
            verbose=False,
            api_results=[([self.get_trade(1, 1), self.get_trade(0, 0)], True, None)],
        )

        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(ts0, ts0 + (self.one_minute * 2))],
            ),
            patch(
                "quant_tick.controllers.rest.get_current_time",
                return_value=ts0 + pd.Timedelta("2h"),
            ),
            patch(
                "quant_tick.controllers.rest.WebSocketData.objects.for_symbol",
            ) as mock_for_symbol,
        ):
            controller.main()

        mock_for_symbol.assert_not_called()
        self.assertEqual(controller.api_calls, 1)

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


@time_machine.travel(datetime(2009, 1, 3), tick=False)
class ExchangeS3Test(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.one_minute = pd.Timedelta("1min")
        self.one_hour = pd.Timedelta("1h")
        self.one_day = pd.Timedelta("1d")
        self.symbol = self.get_symbol()

    def get_data_frame(self) -> pd.DataFrame:
        timestamps = pd.date_range(
            self.timestamp_from,
            self.timestamp_from + self.one_day,
            freq="1min",
            inclusive="left",
        )
        return pd.DataFrame(
            {
                "uid": [str(index) for index, _ in enumerate(timestamps)],
                "timestamp": timestamps,
            }
        )

    def write_existing_day_with_missing_minute(
        self, missing_hour: int = 3, missing_minute: int = 14
    ) -> tuple[datetime, datetime]:
        for hour in range(24):
            ts = self.timestamp_from + (self.one_hour * hour)
            if hour == missing_hour:
                for minute in range(60):
                    if minute == missing_minute:
                        continue
                    TradeData.objects.create(
                        symbol=self.symbol,
                        timestamp=ts + (self.one_minute * minute),
                        frequency=Frequency.MINUTE,
                        ok=True,
                    )
            else:
                TradeData.objects.create(
                    symbol=self.symbol,
                    timestamp=ts,
                    frequency=Frequency.HOUR,
                    ok=True,
                )
        expected_from = (
            self.timestamp_from
            + (self.one_hour * missing_hour)
            + (self.one_minute * missing_minute)
        )
        return expected_from, expected_from + self.one_minute

    def test_main_writes_whole_day_when_no_existing_coverage(self):
        controller = DummyExchangeS3(
            self.symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + self.one_day,
            retry=False,
            verbose=False,
            data_frame=self.get_data_frame(),
        )

        controller.main()

        self.assertEqual(controller.download_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        self.assertEqual(controller.frames[0][0], self.timestamp_from)
        self.assertEqual(controller.frames[0][1], self.timestamp_from + self.one_day)
        self.assertEqual(len(controller.frames[0][2]), 1440)

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 3, 4).replace(tzinfo=UTC),
    )
    def test_main_writes_hours_for_partial_range_with_no_existing_coverage(
        self, mock_get_max_timestamp_to
    ):
        timestamp_to = self.timestamp_from + (self.one_hour * 4)
        controller = DummyExchangeS3(
            self.symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=timestamp_to,
            retry=False,
            verbose=False,
            data_frame=self.get_data_frame(),
        )

        controller.main()

        self.assertEqual(controller.download_calls, 1)
        self.assertEqual(
            [(frame[0], frame[1]) for frame in controller.frames],
            [
                (
                    self.timestamp_from + (self.one_hour * 3),
                    self.timestamp_from + (self.one_hour * 4),
                ),
                (
                    self.timestamp_from + (self.one_hour * 2),
                    self.timestamp_from + (self.one_hour * 3),
                ),
                (
                    self.timestamp_from + self.one_hour,
                    self.timestamp_from + (self.one_hour * 2),
                ),
                (self.timestamp_from, self.timestamp_from + self.one_hour),
            ],
        )
        self.assertTrue(
            all(len(frame[2]) == Frequency.HOUR for frame in controller.frames)
        )

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_main_writes_only_missing_minute_inside_existing_day(
        self, mock_get_max_timestamp_to
    ):
        expected_from, expected_to = self.write_existing_day_with_missing_minute()
        controller = DummyExchangeS3(
            self.symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + self.one_day,
            retry=False,
            verbose=False,
            data_frame=self.get_data_frame(),
        )

        controller.main()

        self.assertEqual(controller.download_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        self.assertEqual(controller.frames[0][0], expected_from)
        self.assertEqual(controller.frames[0][1], expected_to)
        self.assertEqual(list(controller.frames[0][2].timestamp), [expected_from])

    def test_bitmex_main_skips_known_missing_archive_date(self):
        controller = DummyMissingBitmexExchangeS3(
            self.symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + self.one_day,
            retry=False,
            verbose=False,
            data_frame=self.get_data_frame(),
        )

        controller.main()

        self.assertEqual(controller.download_calls, 1)
        self.assertEqual(controller.frames, [])

    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=UTC),
    )
    def test_bitmex_main_writes_only_missing_minute_inside_existing_day(
        self, mock_get_max_timestamp_to
    ):
        expected_from, expected_to = self.write_existing_day_with_missing_minute()
        controller = DummyBitmexExchangeS3(
            self.symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + self.one_day,
            retry=False,
            verbose=False,
            data_frame=self.get_data_frame(),
        )

        controller.main()

        self.assertEqual(controller.download_calls, 1)
        self.assertEqual(len(controller.frames), 1)
        self.assertEqual(controller.frames[0][0], expected_from)
        self.assertEqual(controller.frames[0][1], expected_to)
        self.assertEqual(list(controller.frames[0][2].timestamp), [expected_from])
