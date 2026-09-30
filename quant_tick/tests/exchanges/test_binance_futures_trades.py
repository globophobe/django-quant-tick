from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2
import pandas as pd
import time_machine
from django.test import SimpleTestCase

from quant_tick.constants import RETRY_INDETERMINATE, SymbolType
from quant_tick.exchanges.binance.api import get_binance_api_response
from quant_tick.exchanges.binance_futures.controllers import (
    BinanceFuturesTradesREST,
    BinanceFuturesTradesS3,
    binance_futures_trades,
)
from quant_tick.exchanges.binance_futures.trades import (
    get_binance_futures_trade_pagination_id,
    get_binance_futures_trade_url,
    get_binance_futures_trades,
)
from quant_tick.lib import volume_filter_with_time_window


class BinanceFuturesTradesTest(SimpleTestCase):
    def get_symbol(self):
        return SimpleNamespace(
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )

    def test_rest_uses_public_aggregate_trade_api_without_key(self):
        with patch(
            "quant_tick.exchanges.binance_futures.trades.iter_api",
            return_value=([], False, None),
        ) as mocked:
            get_binance_futures_trades(
                "BTCUSDT",
                datetime(2026, 4, 1, tzinfo=UTC),
                123,
            )

        self.assertEqual(
            mocked.call_args.args[0],
            "https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&limit=1000",
        )
        self.assertEqual(mocked.call_args.kwargs["pagination_id"], 123)
        self.assertEqual(
            get_binance_futures_trade_url(
                "https://example.test?x=1", pagination_id=123
            ),
            "https://example.test?x=1&fromId=123",
        )

    def test_rest_pagination_moves_to_prior_aggregate_ids(self):
        data = [{"a": value} for value in range(2000, 1000, -1)]

        result = get_binance_futures_trade_pagination_id(
            datetime(2026, 4, 1, tzinfo=UTC),
            data=data,
        )

        self.assertEqual(result, 1)

    @time_machine.travel(datetime(2026, 7, 28, tzinfo=UTC), tick=False)
    def test_rest_stops_at_venue_history_limit_without_writing_unfinished_window(self):
        start = datetime(2026, 4, 1, tzinfo=UTC)
        windows = [
            (start + timedelta(hours=hour), start + timedelta(hours=hour + 1))
            for hour in (3, 1, 0)
        ]
        url = "https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&limit=1000"

        def page(first_id, timestamp):
            return httpx2.Response(
                200,
                request=httpx2.Request("GET", url),
                json=[
                    {
                        "a": first_id + index,
                        "p": "100",
                        "q": "1",
                        "f": first_id + index,
                        "l": first_id + index,
                        "T": int(timestamp.timestamp() * 1000) + index,
                        "m": False,
                    }
                    for index in range(1000)
                ],
            )

        for retry in (False, True, RETRY_INDETERMINATE):
            for partial in (False, True):
                with self.subTest(retry=retry, partial=partial):
                    responses = [page(3000, windows[0][0])]
                    if partial:
                        responses.append(
                            page(1000, windows[1][0] + timedelta(minutes=30))
                        )
                    responses.append(
                        httpx2.Response(
                            400,
                            request=httpx2.Request("GET", url),
                            json={
                                "code": -4166,
                                "msg": "Search window is restricted to recent 2 days only.",
                            },
                        )
                    )
                    persisted = Mock()
                    controller = BinanceFuturesTradesREST(
                        self.get_symbol(), start, windows[0][1], persisted, retry=retry
                    )
                    controller.get_candles = Mock(return_value=pd.DataFrame([]))
                    controller.get_pagination_id = Mock(side_effect=[3000, 1000])
                    with (
                        patch(
                            "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                            return_value=windows,
                        ) as iterator,
                        patch(
                            "quant_tick.exchanges.binance.api.httpx2.get",
                            side_effect=responses,
                        ) as get,
                        patch("quant_tick.exchanges.binance.api.time.sleep") as sleep,
                    ):
                        controller.main()

                    iterator.assert_called_once_with(start, windows[0][1], retry=retry)
                    self.assertEqual(get.call_count, len(responses))
                    sleep.assert_not_called()
                    self.assertEqual(controller.get_candles.call_count, 2)
                    persisted.assert_called_once()
                    self.assertEqual(persisted.call_args.args[1:3], windows[0])
                    self.assertEqual(
                        persisted.call_args.args[3]["uid"].tolist(),
                        [str(value) for value in range(3000, 4000)],
                    )

    def test_rest_does_not_treat_other_http_errors_as_history_exhaustion(self):
        for status, path, body in (
            (400, "/fapi/v1/aggTrades", b'{"code": -1121}'),
            (400, "/fapi/v1/aggTrades", b"not JSON"),
            (400, "/fapi/v1/aggTrades", b"[]"),
            (400, "/fapi/v1/klines", b'{"code": -4166}'),
            (400, "/api/v3/aggTrades", b'{"code": -4166}'),
            (500, "/fapi/v1/aggTrades", b'{"code": -4166}'),
        ):
            url = f"https://fapi.binance.com{path}?symbol=BTCUSDT"
            response = httpx2.Response(
                status, request=httpx2.Request("GET", url), content=body
            )
            with (
                self.subTest(status=status, path=path, body=body),
                patch(
                    "quant_tick.exchanges.binance.api.httpx2.get",
                    return_value=response,
                ) as get,
                patch("quant_tick.exchanges.binance.api.time.sleep"),
                self.assertRaises(httpx2.HTTPStatusError),
            ):
                get_binance_api_response(get_binance_futures_trade_url, url, retry=1)
            self.assertEqual(get.call_count, 2)

    def test_rest_normalizes_aggregate_trade_schema(self):
        controller = BinanceFuturesTradesREST.__new__(BinanceFuturesTradesREST)
        rows = controller.parse_data(
            [
                {
                    "a": 12,
                    "p": "100",
                    "q": "1",
                    "f": 23,
                    "l": 25,
                    "T": 1775606403000,
                    "m": False,
                },
                {
                    "a": 11,
                    "p": "100",
                    "q": "20",
                    "f": 3,
                    "l": 22,
                    "T": 1775606402000,
                    "m": False,
                },
                {
                    "a": 10,
                    "p": "100",
                    "q": "2",
                    "f": 1,
                    "l": 2,
                    "T": 1775606401000,
                    "m": True,
                },
            ]
        )

        aggregated = controller.get_data_frame(rows)
        filtered = volume_filter_with_time_window(aggregated, min_volume=1000)

        self.assertEqual(aggregated["uid"].tolist(), ["10", "11", "12"])
        self.assertEqual(aggregated["tickRule"].tolist(), [-1, 1, 1])
        self.assertEqual(
            aggregated["notional"].tolist(), [Decimal(2), Decimal(20), Decimal(1)]
        )
        self.assertEqual(
            aggregated["volume"].tolist(),
            [Decimal(200), Decimal(2000), Decimal(100)],
        )
        self.assertEqual(aggregated["ticks"].tolist(), [2, 20, 3])
        self.assertEqual(filtered["uid"].tolist(), ["11", "12"])
        self.assertEqual(filtered.iloc[0]["volume"], Decimal(2000))
        self.assertEqual(filtered.iloc[0]["totalVolume"], Decimal(2200))
        self.assertEqual(filtered.iloc[0]["totalBuyVolume"], Decimal(2000))
        self.assertEqual(filtered.iloc[0]["totalTicks"], 22)
        self.assertEqual(filtered.iloc[0]["totalBuyTicks"], 20)
        self.assertTrue(pd.isna(filtered.iloc[1]["volume"]))
        self.assertEqual(filtered.iloc[1]["totalVolume"], Decimal(100))
        self.assertEqual(filtered.iloc[1]["totalTicks"], 3)

    def test_s3_uses_aggregate_trade_archive(self):
        controller = BinanceFuturesTradesS3.__new__(BinanceFuturesTradesS3)
        controller.symbol = self.get_symbol()

        self.assertEqual(
            controller.get_url(date(2026, 7, 20)),
            "https://data.binance.vision/data/futures/um/daily/aggTrades/"
            "BTCUSDT/BTCUSDT-aggTrades-2026-07-20.zip",
        )

        timestamp_from = datetime(2026, 7, 20, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(days=1)
        with patch(
            "quant_tick.exchanges.binance_futures.base.binance_futures_candles",
            return_value=pd.DataFrame([]),
        ) as mocked:
            controller.get_candles(timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            "BTCUSDT",
            timestamp_from,
            timestamp_to,
            interval="1m",
        )

    def test_s3_normalizes_chunked_archive_schema(self):
        first_hour = datetime(2026, 7, 20, tzinfo=UTC)
        second_hour = first_hour + timedelta(hours=1)
        timestamp_to = second_hour + timedelta(hours=1)
        data = pd.DataFrame(
            [
                {
                    "agg_trade_id": "agg_trade_id",
                    "price": "price",
                    "quantity": "quantity",
                    "first_trade_id": "first_trade_id",
                    "last_trade_id": "last_trade_id",
                    "transact_time": "transact_time",
                    "is_buyer_maker": "is_buyer_maker",
                },
                {
                    "agg_trade_id": "3387637573",
                    "price": "64694.8",
                    "quantity": "0.597",
                    "first_trade_id": "7912146392",
                    "last_trade_id": "7912146438",
                    "transact_time": "1784505600061",
                    "is_buyer_maker": "false",
                },
                {
                    "agg_trade_id": "3387637574",
                    "price": "64700",
                    "quantity": "0.100",
                    "first_trade_id": "7912146439",
                    "last_trade_id": "7912146440",
                    "transact_time": "1784509200061",
                    "is_buyer_maker": "true",
                },
            ]
        )
        symbol = self.get_symbol()
        on_data_frame = Mock()
        controller = BinanceFuturesTradesS3(
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
        self.assertEqual(first.iloc[0].uid, "3387637573")
        self.assertEqual(first.iloc[0].notional, Decimal("0.597"))
        self.assertEqual(first.iloc[0].tickRule, 1)
        self.assertEqual(first.iloc[0].ticks, 47)
        self.assertEqual(
            first.iloc[0].timestamp,
            pd.Timestamp("2026-07-20T00:00:00.061Z"),
        )
        self.assertEqual(first.iloc[0].nanoseconds, 0)
        self.assertEqual(second["uid"].tolist(), ["3387637574"])

    def test_s3_keeps_microseconds_in_timestamp_not_residual_nanoseconds(self):
        controller = BinanceFuturesTradesS3.__new__(BinanceFuturesTradesS3)
        data = pd.DataFrame(
            [
                {
                    "agg_trade_id": "1",
                    "price": "64694.8",
                    "quantity": "0.597",
                    "first_trade_id": "10",
                    "last_trade_id": "10",
                    "transact_time": "1784505600061123",
                    "is_buyer_maker": "false",
                }
            ]
        )

        parsed = controller.parse_dtypes_and_strip_columns(data)

        self.assertEqual(
            parsed.iloc[0].timestamp,
            pd.Timestamp("2026-07-20T00:00:00.061123Z"),
        )
        self.assertEqual(parsed.iloc[0].nanoseconds, 0)

    @time_machine.travel(datetime(2026, 7, 28, 12, 34, 56, tzinfo=UTC), tick=False)
    def test_controller_bounds_recent_followup_and_persists_aggregates(self):
        timestamp_to = datetime(2026, 7, 28, 12, 34, tzinfo=UTC)
        cutoff = datetime(2026, 7, 21, 12, 34, tzinfo=UTC)
        old_start = datetime(2019, 12, 30, tzinfo=UTC)
        recent_start = timestamp_to - timedelta(days=2)
        symbol = self.get_symbol()
        data_frame = pd.DataFrame([{"uid": "1"}])
        candles = pd.DataFrame([])

        for start, end, retry, expected_start in (
            (old_start, timestamp_to, False, cutoff),
            (old_start, timestamp_to, True, cutoff),
            (old_start, timestamp_to, RETRY_INDETERMINATE, cutoff),
            (recent_start, timestamp_to, False, recent_start),
            (cutoff, cutoff + timedelta(minutes=1), False, cutoff),
            (old_start, cutoff, False, None),
            (old_start, cutoff - timedelta(days=1), True, None),
            (old_start, timestamp_to - timedelta(days=4), False, cutoff),
        ):
            calls = []
            on_data_frame = Mock()
            with (
                self.subTest(start=start, end=end, retry=retry),
                patch(
                    "quant_tick.exchanges.binance_futures.controllers."
                    "BinanceFuturesTradesS3"
                ) as archive,
                patch(
                    "quant_tick.exchanges.binance_futures.controllers."
                    "BinanceFuturesTradesREST"
                ) as rest,
            ):
                archive.return_value.main.side_effect = lambda calls=calls: (
                    calls.append("archive")
                )
                rest.return_value.main.side_effect = lambda calls=calls: calls.append(
                    "rest"
                )
                binance_futures_trades(
                    symbol, start, end, on_data_frame, retry=retry, verbose=True
                )

                callback = archive.call_args.kwargs["on_data_frame"]
                kwargs = {
                    "timestamp_from": start,
                    "timestamp_to": end,
                    "on_data_frame": callback,
                    "retry": retry,
                    "verbose": True,
                }
                archive.assert_called_once_with(symbol, **kwargs)
                if expected_start is None:
                    self.assertEqual(calls, ["archive"])
                    rest.assert_not_called()
                else:
                    self.assertEqual(calls, ["archive", "rest"])
                    kwargs["timestamp_from"] = expected_start
                    rest.assert_called_once_with(symbol, **kwargs)
                callback(symbol, start, end, data_frame, candles)
                on_data_frame.assert_called_once_with(
                    symbol,
                    start,
                    end,
                    data_frame,
                    candles,
                    aggregated_trades=data_frame,
                )
