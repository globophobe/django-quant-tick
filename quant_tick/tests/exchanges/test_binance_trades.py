from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import time_machine
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import Exchange, Frequency
from quant_tick.exchanges.api import api
from quant_tick.exchanges.binance.controllers import (
    BinanceTradesREST,
    BinanceTradesS3,
)
from quant_tick.exchanges.binance.trades import get_trades
from quant_tick.lib import get_existing
from quant_tick.models import TradeData

from ..base import BaseSymbolTest


class BinanceTradesTest(SimpleTestCase):
    def test_get_trades_uses_spot_raw_trade_api(self):
        with patch(
            "quant_tick.exchanges.binance.trades.iter_api",
            return_value=([], False, None),
        ) as mocked:
            get_trades("BTCUSDT", datetime(2026, 4, 1, tzinfo=UTC), 1)

        url = mocked.call_args.args[0]
        self.assertTrue(
            url.startswith("https://api.binance.com/api/v3/historicalTrades?")
        )

    def test_s3_uses_spot_raw_trade_archive(self):
        controller = BinanceTradesS3.__new__(BinanceTradesS3)
        controller.symbol = SimpleNamespace(api_symbol="BTCUSDT")

        url = controller.get_url(date(2026, 4, 1))

        self.assertEqual(
            url,
            "https://data.binance.vision/data/spot/daily/trades/"
            "BTCUSDT/BTCUSDT-trades-2026-04-01.zip",
        )

    def test_s3_normalizes_chunked_hours_and_tick_rules(self):
        first_hour = datetime(2026, 4, 8, tzinfo=UTC)
        second_hour = first_hour + timedelta(hours=1)
        timestamp_to = second_hour + timedelta(hours=1)
        data = pd.DataFrame(
            [
                {
                    "id": "id",
                    "price": "price",
                    "qty": "qty",
                    "quoteQty": "quote_qty",
                    "time": "time",
                    "isBuyerMaker": "is_buyer_maker",
                    "isBestMatch": None,
                },
                {
                    "id": "1",
                    "price": "100",
                    "qty": "2",
                    "quoteQty": "200",
                    "time": "1775606400000",
                    "isBuyerMaker": True,
                    "isBestMatch": True,
                },
                {
                    "id": "2",
                    "price": "101",
                    "qty": "3",
                    "quoteQty": "303",
                    "time": "1775606401000",
                    "isBuyerMaker": "True",
                    "isBestMatch": True,
                },
                {
                    "id": "3",
                    "price": "102",
                    "qty": "4",
                    "quoteQty": "408",
                    "time": "1775606402000",
                    "isBuyerMaker": False,
                    "isBestMatch": True,
                },
                {
                    "id": "4",
                    "price": "103",
                    "qty": "5",
                    "quoteQty": "515",
                    "time": "1775606403000",
                    "isBuyerMaker": "False",
                    "isBestMatch": True,
                },
                {
                    "id": "5",
                    "price": "104",
                    "qty": "6",
                    "quoteQty": "624",
                    "time": "1775610000000",
                    "isBuyerMaker": False,
                    "isBestMatch": True,
                },
            ]
        )
        symbol = SimpleNamespace(api_symbol="BTCUSDT")
        on_data_frame = Mock()
        controller = BinanceTradesS3(
            symbol,
            timestamp_from=first_hour,
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
        )
        controller.get_data_frame_chunks = Mock(
            return_value=iter([data.iloc[:2].copy(), data.iloc[2:].copy()])
        )
        controller.get_candles = Mock(return_value=pd.DataFrame([]))

        with (
            patch(
                "quant_tick.controllers.s3.TradeDataIterator.iter_days",
                return_value=[(first_hour, timestamp_to, [])],
            ),
            patch(
                "quant_tick.controllers.s3.TradeDataIterator.iter_hours",
                return_value=[
                    (second_hour, timestamp_to),
                    (first_hour, second_hour),
                ],
            ),
        ):
            controller.main()

        self.assertEqual(
            [(call.args[1], call.args[2]) for call in on_data_frame.call_args_list],
            [
                (first_hour, second_hour),
                (second_hour, timestamp_to),
            ],
        )
        first = on_data_frame.call_args_list[0].args[3]
        second = on_data_frame.call_args_list[1].args[3]
        self.assertEqual(first["uid"].tolist(), ["1", "2", "3", "4"])
        self.assertEqual(first["tickRule"].tolist(), [-1, -1, 1, 1])
        self.assertEqual(second["uid"].tolist(), ["5"])


@time_machine.travel(datetime(2026, 7, 28, 12, 34, 56, tzinfo=UTC), tick=False)
@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class BinanceTradeCollectionTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.day = datetime(2026, 7, 21, tzinfo=UTC)
        timestamps = pd.date_range(
            self.day, self.day + timedelta(days=1), freq="1min", inclusive="left"
        )
        self.trades = [
            {
                "id": index + 1,
                "time": int(timestamp.timestamp() * 1000),
                "price": "100",
                "qty": "1",
                "isBuyerMaker": False,
            }
            for index, timestamp in enumerate(timestamps)
        ]
        self.candles = pd.DataFrame({"notional": Decimal(1)}, index=timestamps)

    def test_followup_preserves_complete_daily_and_hourly_partitions(self):
        end = self.day + timedelta(days=1)
        for frequency in (Frequency.DAY, Frequency.HOUR):
            with self.subTest(frequency=frequency):
                symbol = self.get_symbol(
                    exchange=Exchange.BINANCE, api_symbol="BTCUSDT"
                )
                controller = BinanceTradesREST(symbol, self.day, end, Mock())
                raw = controller.get_data_frame(
                    controller.parse_data(list(reversed(self.trades)))
                )
                partitions = []
                for minute in range(0, Frequency.DAY, frequency):
                    start = self.day + timedelta(minutes=minute)
                    partitions.extend(
                        TradeData.write(
                            symbol,
                            start,
                            start + timedelta(minutes=frequency),
                            self.candles,
                            raw_trades=raw,
                        )
                    )
                self.assertTrue(all(partition.ok for partition in partitions))

                with (
                    patch(
                        "quant_tick.exchanges.binance.controllers.zip_chunk_downloader",
                        return_value=None,
                    ),
                    patch(
                        "quant_tick.exchanges.binance.base.binance_candles",
                        return_value=self.candles,
                    ),
                    patch(
                        "quant_tick.exchanges.binance.base.get_trades",
                        return_value=(list(reversed(self.trades)), True, None),
                    ) as rest,
                ):
                    api(symbol, self.day - timedelta(days=30), end)

                rows = TradeData.objects.filter(symbol=symbol)
                covered = get_existing(rows.values("timestamp", "frequency"))
                self.assertEqual(len(set(covered)), Frequency.DAY)
                self.assertEqual(
                    list(rows.values_list("pk", flat=True)),
                    [partition.pk for partition in partitions],
                )
                for partition in partitions:
                    self.assertTrue(
                        partition.raw_data.storage.exists(partition.raw_data.name)
                    )
                rest.assert_not_called()

    def test_followup_repairs_first_day_with_partial_request_end(self):
        symbol = self.get_symbol(exchange=Exchange.BINANCE, api_symbol="BTCUSDT")
        end = datetime(2026, 7, 28, 12, 34, tzinfo=UTC)
        gap = self.day + timedelta(hours=20, minutes=10)
        gap_hour = gap.replace(minute=0)
        last_hour = end.replace(minute=0)
        partitions = [
            TradeData(
                symbol=symbol, timestamp=timestamp, frequency=Frequency.HOUR, ok=True
            )
            for timestamp in pd.date_range(
                self.day, last_hour, freq="1h", inclusive="left"
            )
            if timestamp != gap_hour
        ]
        for start, stop in (
            (gap_hour, gap_hour + timedelta(hours=1)),
            (last_hour, end),
        ):
            partitions.extend(
                TradeData(
                    symbol=symbol,
                    timestamp=timestamp,
                    frequency=Frequency.MINUTE,
                    ok=True,
                )
                for timestamp in pd.date_range(
                    start, stop, freq="1min", inclusive="left"
                )
                if timestamp != gap
            )
        TradeData.objects.bulk_create(partitions)
        self.assertFalse(TradeData.objects.has_timestamps(symbol, self.day, end))
        trade = self.trades[20 * 60 + 10]

        with (
            patch(
                "quant_tick.exchanges.binance.controllers.zip_chunk_downloader",
                return_value=None,
            ),
            patch(
                "quant_tick.exchanges.binance.base.binance_candles",
                return_value=self.candles,
            ),
            patch(
                "quant_tick.exchanges.binance.base.get_trades",
                return_value=([trade], True, None),
            ) as rest,
        ):
            api(symbol, self.day - timedelta(days=30), end)

        self.assertTrue(TradeData.objects.has_timestamps(symbol, self.day, end))
        repaired = TradeData.objects.get(symbol=symbol, timestamp=gap)
        self.assertEqual(repaired.uid, str(trade["id"]))
        self.assertIs(repaired.ok, True)
        self.assertEqual(repaired.get_data_frame("raw_data").iloc[0].timestamp, gap)
        rest.assert_called_once()
