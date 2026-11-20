from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd
import time_machine
from django.test import TestCase, override_settings

from quant_tick.constants import RETRY_INDETERMINATE, FileData, Frequency
from quant_tick.controllers import ExchangeS3
from quant_tick.lib.download import ArchiveDownloadError
from quant_tick.models import TradeData, WebSocketData

from ..base import BaseSymbolTest
from .test_rest import DummyExchangeREST, DummyExchangeWebSocket


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


@time_machine.travel(datetime(2009, 1, 3, tzinfo=UTC), tick=False)
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

    @override_settings(
        STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
    )
    def test_websocket_prepass_preserves_archive_fallback_for_unvalidated_minutes(self):
        start, end = self.timestamp_from, self.timestamp_from + self.one_minute
        trade = {
            "uid": "archive",
            "timestamp": start,
            "nanoseconds": 0,
            "price": Decimal(100),
            "volume": Decimal(100),
            "notional": Decimal(1),
            "tickRule": 1,
            "index": 0,
        }

        def persist(symbol, lower, upper, frame, candles, **kwargs):
            if not kwargs:
                kwargs["raw_trades"] = frame
            return TradeData.write(symbol, lower, upper, candles, **kwargs)

        for mode in (DummyExchangeREST, DummyExchangeWebSocket):
            for retry in (False, RETRY_INDETERMINATE):
                for scenario in ("valid", "invalid", "unknown", "quiet"):
                    with self.subTest(
                        mode=mode.__name__, retry=retry, scenario=scenario
                    ):
                        TradeData.objects.all().delete()
                        WebSocketData.objects.all().delete()
                        if retry:
                            TradeData.objects.create(
                                symbol=self.symbol,
                                timestamp=start,
                                frequency=Frequency.MINUTE,
                                ok=None,
                            )
                        if scenario != "quiet":
                            quantity = Decimal("0.5" if scenario == "invalid" else "1")
                            WebSocketData.objects.create(
                                exchange=self.symbol.exchange,
                                api_symbol=self.symbol.api_symbol,
                                significant_trade_filter=0,
                                timestamp=start,
                                raw_trades=[
                                    {
                                        **trade,
                                        "uid": "websocket",
                                        "price": Decimal(101),
                                        "notional": quantity,
                                        "volume": quantity * 101,
                                    }
                                ],
                            )
                        candles = (
                            pd.DataFrame([])
                            if scenario == "unknown"
                            else pd.DataFrame(
                                {
                                    "notional": [
                                        Decimal(0 if scenario == "quiet" else 1)
                                    ]
                                },
                                index=[start],
                            )
                        )
                        native = mode(
                            self.symbol, start, end, retry=retry, api_results=[]
                        )
                        archive = DummyExchangeS3(
                            self.symbol,
                            start,
                            end,
                            retry=retry,
                            data_frame=pd.DataFrame([trade]),
                        )
                        native.on_data_frame = archive.on_data_frame = persist
                        with (
                            time_machine.travel(end + timedelta(minutes=3), tick=False),
                            patch.object(native, "get_candles", return_value=candles),
                            patch.object(archive, "get_candles", return_value=candles),
                        ):
                            native.collect_websocket()
                            archive.main()
                        stored = TradeData.objects.get(symbol=self.symbol)
                        use_archive = scenario in ("invalid", "unknown")
                        self.assertEqual(archive.download_calls, int(use_archive))
                        self.assertIs(
                            stored.ok, None if scenario == "unknown" else True
                        )
                        frame = stored.get_data_frame(FileData.RAW)
                        if scenario == "quiet":
                            self.assertIsNone(frame)
                        else:
                            self.assertEqual(frame.notional.sum(), Decimal(1))
                            self.assertEqual(
                                frame.price.tolist(),
                                [Decimal(100 if use_archive else 101)],
                            )

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
        return_value=datetime(2009, 1, 3, 4, tzinfo=UTC),
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
        return_value=datetime(2009, 1, 4, tzinfo=UTC),
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

    @time_machine.travel(datetime(2009, 1, 10, tzinfo=UTC), tick=False)
    @patch(
        "quant_tick.controllers.iterators.TradeDataIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4, tzinfo=UTC),
    )
    def test_main_skips_leading_unpublished_days_then_stops_inside_history(
        self, mock_get_max_timestamp_to
    ):
        timestamp_from = datetime(2009, 1, 1, tzinfo=UTC)
        timestamp_to = datetime(2009, 1, 4, tzinfo=UTC)
        available = self.get_data_frame().copy()
        available["timestamp"] = available["timestamp"] - self.one_day
        for unavailable in (None, ArchiveDownloadError("HTTP 403", status_code=403)):
            with self.subTest(unavailable=unavailable):
                controller = DummyExchangeS3(
                    self.symbol,
                    timestamp_from=timestamp_from,
                    timestamp_to=timestamp_to,
                    retry=False,
                    verbose=False,
                    data_frame=available,
                )
                controller.get_data_frame = Mock(
                    side_effect=[unavailable, available, None]
                )

                controller.main()

                self.assertEqual(
                    [call.args[0] for call in controller.get_data_frame.call_args_list],
                    [
                        datetime(2009, 1, 3, tzinfo=UTC).date(),
                        datetime(2009, 1, 2, tzinfo=UTC).date(),
                        datetime(2009, 1, 1, tzinfo=UTC).date(),
                    ],
                )
                self.assertEqual(len(controller.frames), 1)
                self.assertEqual(
                    controller.frames[0][0], datetime(2009, 1, 2, tzinfo=UTC)
                )
