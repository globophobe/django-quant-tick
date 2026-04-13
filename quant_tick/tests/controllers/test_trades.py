from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import Frequency
from quant_tick.controllers import ExchangeREST, TradeDataIterator
from quant_tick.models import TradeData

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
        self.api_results = api_results
        self.api_calls = 0
        self.pagination_ids = []
        super().__init__(*args, on_data_frame=self.capture_frame, **kwargs)

    def capture_frame(self, symbol, timestamp_from, timestamp_to, data_frame, candles):
        self.frames.append((timestamp_from, timestamp_to, data_frame.copy()))

    def get_pagination_id(self, timestamp_to: datetime) -> str:
        return timestamp_to.isoformat()

    def iter_api(self, timestamp_from: datetime, pagination_id: str) -> tuple:
        self.pagination_ids.append(pagination_id)
        result = self.api_results[self.api_calls]
        self.api_calls += 1
        return result

    def parse_data(self, data: list) -> list:
        return data

    def get_candles(self, timestamp_from: datetime, timestamp_to: datetime) -> pd.DataFrame:
        return pd.DataFrame([])


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
