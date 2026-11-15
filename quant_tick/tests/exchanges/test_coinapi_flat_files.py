import csv
import gzip
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2
import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.coinapi.controllers import CoinAPITrades
from quant_tick.exchanges.coinapi.flat_files import (
    BASE_URL,
    CSV_COLUMNS,
    CoinAPITradeFiles,
    find_trade_file,
    get_flat_file_content,
)
from quant_tick.exchanges.hyperliquid.controllers import HyperliquidTrades
from quant_tick.lib.download import ArchiveDownloadError
from quant_tick.models import Symbol, TradeData

START = datetime(2026, 11, 13, 12, tzinfo=UTC)
SYMBOL = "HYPERLIQUID_PERP_BTC_USDC"
KEY = "T-TRADES/D-2026111312/E-HYPERLIQUID/IDDI-42+SC-" + SYMBOL + "+S-BTC.csv.gz"
MODULE = "quant_tick.exchanges.coinapi.flat_files."


def listing(keys=(KEY,), *, truncated="false", token=""):
    contents = "".join("<Contents><Key>" + key + "</Key></Contents>" for key in keys)
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        + contents
        + "<IsTruncated>"
        + truncated
        + "</IsTruncated><NextContinuationToken>"
        + token
        + "</NextContinuationToken></ListBucketResult>"
    ).encode()


def archive(rows):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, delimiter=";")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(output.getvalue().encode())


def trade(uid="a", minute=0, **kwargs):
    return {
        "time_exchange": (START + timedelta(minutes=minute)).isoformat(),
        "time_coinapi": (START + timedelta(minutes=minute, seconds=1)).isoformat(),
        "guid": uid,
        "price": "100.123456789123456789",
        "base_amount": "1",
        "taker_side": "BUY",
        "id_exch_guid": "",
        "id_exch_int_inc": "106950518024024",
        **kwargs,
    }


def response(content, status=200):
    return httpx2.Response(
        status, content=content, request=httpx2.Request("GET", BASE_URL)
    )


@patch.dict(os.environ, {"COINAPI_API_KEY": "test-key"}, clear=True)
class CoinAPIFlatFilesTest(SimpleTestCase):
    def setUp(self):
        for target in ("throttle_api_requests", "increment_api_total_requests"):
            patcher = patch(MODULE + target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_file_is_cached_and_preserves_order_precision_ids_and_direction(self):
        rows = [
            trade("late", minute=1),
            trade("second", time_exchange="2026-11-13T12:00:00.0000002"),
            trade(
                "first", time_exchange="2026-11-13T12:00:00.0000001", taker_side="SELL"
            ),
            trade("outside", minute=-1),
        ]
        files = CoinAPITradeFiles(SYMBOL)
        with (
            patch(
                MODULE + "httpx2.get",
                side_effect=[response(listing()), response(archive(rows))],
            ) as request,
            patch(MODULE + "pd.read_csv", wraps=pd.read_csv) as parse,
        ):
            selected = files.get_trades(START, START + timedelta(minutes=1))
            later = files.get_trades(
                START + timedelta(minutes=1), START + timedelta(minutes=2)
            )
        self.assertEqual(request.call_count, 2)
        parse.assert_called_once()
        self.assertEqual(
            request.call_args.kwargs["headers"]["Authorization"], "test-key"
        )
        self.assertIn("%2BSC-" + SYMBOL, request.call_args.args[0])
        self.assertNotIn("test-key", request.call_args.args[0])
        self.assertEqual([row["uuid"] for row in selected], ["first", "second"])
        self.assertEqual([row["uuid"] for row in later], ["late"])
        controller = CoinAPITrades(
            SimpleNamespace(api_symbol=SYMBOL), START, START, Mock()
        )
        frame = controller.get_data_frame(controller.parse_data(selected[::-1]))
        self.assertEqual(frame.nanoseconds.tolist(), [100, 200])
        self.assertEqual(frame.tickRule.tolist(), [-1, 1])
        self.assertEqual(frame.volume.sum(), Decimal("200.246913578246913578"))
        self.assertEqual(frame.id_trade.tolist(), ["106950518024024"] * 2)
        self.assertEqual(frame.coinapi_transport.tolist(), ["flat_files"] * 2)

    def test_replayed_and_incomplete_minutes_choose_file_or_rest_exclusively(self):
        candles = pd.DataFrame(
            {
                "notional": [Decimal("2.0000000000000002"), Decimal(2)],
                "data_source": "coinapi",
                "trades_count": [2, 2],
            },
            index=[START, START + timedelta(minutes=1)],
        )
        rest_rows = [
            {
                "uuid": "rest-" + str(minute),
                "time_exchange": (START + timedelta(minutes=minute)).isoformat(),
                "time_coinapi": (
                    START + timedelta(minutes=minute, seconds=1)
                ).isoformat(),
                "price": "100",
                "size": "2",
                "taker_side": "SELL",
            }
            for minute in (1, 2)
        ]
        for controller_type, exchange, api_symbol, rows in (
            (CoinAPITrades, Exchange.COINAPI, SYMBOL, rest_rows),
            (CoinAPITrades, Exchange.COINAPI, SYMBOL, []),
            (HyperliquidTrades, Exchange.HYPERLIQUID, "BTC", rest_rows),
            (HyperliquidTrades, Exchange.HYPERLIQUID, "BTC", []),
        ):
            symbol = Symbol(
                exchange=exchange,
                api_symbol=api_symbol,
                symbol_type=SymbolType.PERPETUAL,
                save_raw=True,
            )
            callback = Mock()
            end = START + timedelta(minutes=3)
            controller = controller_type(symbol, START, end, callback)
            with (
                self.subTest(exchange=exchange, empty_rest=not rows),
                patch(
                    MODULE + "httpx2.get",
                    side_effect=[
                        response(listing()),
                        response(
                            archive(
                                [
                                    trade(
                                        "file-0",
                                        time_exchange="2026-11-13T12:00:00.0000001Z",
                                    ),
                                    trade(
                                        "file-0-replay",
                                        time_exchange="2026-11-13T12:00:00.0000001Z",
                                        time_coinapi="2026-11-13T12:00:30Z",
                                    ),
                                    trade("incomplete", minute=1),
                                ]
                            )
                        ),
                    ],
                ),
                patch(
                    "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                    return_value=rows,
                ) as rest,
            ):
                frame = controller.on_rest_data_frame(START, end, candles)
                rest.assert_called_once()
                self.assertEqual(
                    pd.Timestamp(rest.call_args.args[1]["time_start"]),
                    START + timedelta(minutes=1),
                )
                self.assertEqual(
                    frame.uid.tolist(),
                    ["file-0", "rest-1", "rest-2"] if rows else ["file-0"],
                )
                self.assertEqual(
                    frame.coinapi_replayed_notional.tolist(),
                    [Decimal(1), 0, 0] if rows else [Decimal(1)],
                )
                self.assertEqual(
                    candles.iloc[0].notional, Decimal("2.0000000000000002")
                )
                self.assertEqual(
                    frame.coinapi_transport.tolist(),
                    ["flat_files", "rest", "rest"] if rows else ["flat_files"],
                )
                self.assertEqual(
                    frame.notional.sum(), Decimal(5) if rows else Decimal(1)
                )
                self.assertEqual(frame.timestamp.dt.nanosecond.iloc[0], 100)
                self.assertEqual(
                    frame.timestamp.dt.floor("min").tolist(),
                    [START + timedelta(minutes=i) for i in range(3)]
                    if rows
                    else [START],
                )
                self.assertIs(
                    TradeData.validate(
                        symbol, START, end, callback.call_args.args[4], raw_trades=frame
                    ),
                    None if rows else False,
                )

    def test_rounding_tolerance_preserves_zero_and_missing_trade_checks(self):
        for quantity, expected, needs_rest in (
            ("1", "0.9999999999999999", False),
            ("1", "1.00000002", True),
            (None, "0", False),
            (None, "0.000000001", True),
            ("0.000000001", "0", True),
        ):
            controller = CoinAPITrades(
                SimpleNamespace(api_symbol=SYMBOL), START, START, Mock()
            )
            candles = pd.DataFrame({"notional": [Decimal(expected)]}, index=[START])
            rows = [] if quantity is None else [trade(base_amount=quantity)]
            with (
                self.subTest(quantity=quantity, expected=expected),
                patch(
                    MODULE + "httpx2.get",
                    side_effect=[response(listing()), response(archive(rows))],
                ),
                patch.object(
                    controller,
                    "fetch_rest_data_frame",
                    return_value=controller.get_data_frame([]),
                ) as rest,
            ):
                controller.fetch_trade_data_frame(
                    START, START + timedelta(minutes=1), candles
                )
            self.assertEqual(rest.called, needs_rest)

    def test_replays_only_adjust_explained_coinapi_candle_volume(self):
        for source, quantity, count, needs_rest, adjusted in (
            ("coinapi", "2", 2, False, "1"),
            ("hyperliquid", "2", 2, True, "2"),
            (None, "2", 2, True, "2"),
            ("coinapi", "2.5", 2, True, "2.5"),
            ("coinapi", "2", 3, True, "2"),
            ("coinapi", "2", None, True, "2"),
            ("coinapi", "1", 2, False, "1"),
            ("hyperliquid", "1", 2, False, "1"),
        ):
            callback = Mock()
            controller = CoinAPITrades(
                SimpleNamespace(api_symbol=SYMBOL), START, START, callback
            )
            candles = pd.DataFrame(
                {
                    "notional": [Decimal(quantity)],
                    "data_source": source,
                    "trades_count": count,
                },
                index=[START],
            )
            rows = [trade("original"), trade("replay")]
            with (
                self.subTest(source=source, quantity=quantity),
                patch(
                    MODULE + "httpx2.get",
                    side_effect=[response(listing()), response(archive(rows))],
                ),
                patch.object(
                    controller,
                    "fetch_rest_data_frame",
                    return_value=controller.get_data_frame([]),
                ) as rest,
            ):
                controller.on_rest_data_frame(
                    START, START + timedelta(minutes=1), candles
                )
            self.assertEqual(rest.called, needs_rest)
            self.assertEqual(
                callback.call_args.args[4].iloc[0].notional, Decimal(adjusted)
            )
            self.assertEqual(candles.iloc[0].notional, Decimal(quantity))

    def test_missing_object_falls_back_and_access_or_corruption_errors_propagate(self):
        candles = pd.DataFrame({"notional": [Decimal(1)]}, index=[START])
        for status in (404, 550, 401, 402, 403):
            controller = CoinAPITrades(
                SimpleNamespace(api_symbol=SYMBOL),
                START,
                START + timedelta(minutes=1),
                Mock(),
            )
            with (
                self.subTest(status=status),
                patch(
                    MODULE + "httpx2.get", return_value=response(b"missing", status)
                ) as request,
                patch.object(
                    controller, "fetch_rest_data_frame", return_value=pd.DataFrame()
                ) as rest,
            ):
                if status in (404, 550):
                    controller.fetch_trade_data_frame(
                        START, START + timedelta(minutes=1), candles
                    )
                    rest.assert_called_once()
                else:
                    with self.assertRaises(ArchiveDownloadError) as error:
                        controller.fetch_trade_data_frame(
                            START, START + timedelta(minutes=1), candles
                        )
                    self.assertEqual(error.exception.status_code, status)
                    rest.assert_not_called()
                request.assert_called_once()
        with (
            patch(
                MODULE + "httpx2.get",
                side_effect=[response(listing()), response(b"not gzip")],
            ),
            self.assertRaises(gzip.BadGzipFile),
        ):
            CoinAPITradeFiles(SYMBOL).get_trades(START, START + timedelta(minutes=1))

    def test_open_hour_is_not_requested_and_missing_file_is_cached_only_for_run(self):
        files = CoinAPITradeFiles(SYMBOL)
        with (
            patch(
                MODULE + "get_current_time", return_value=START + timedelta(minutes=30)
            ),
            patch(MODULE + "httpx2.get") as request,
        ):
            self.assertIsNone(files.get_trades(START, START + timedelta(minutes=1)))
            request.assert_not_called()
        with patch(
            MODULE + "httpx2.get", return_value=response(listing(keys=()))
        ) as request:
            self.assertIsNone(files.get_trades(START, START + timedelta(minutes=1)))
            self.assertIsNone(files.get_trades(START, START + timedelta(minutes=2)))
            self.assertEqual(request.call_count, 1)
            self.assertIsNone(
                CoinAPITradeFiles(SYMBOL).get_trades(
                    START, START + timedelta(minutes=1)
                )
            )
            self.assertEqual(request.call_count, 2)

    def test_legacy_day_and_new_hour_paths_and_no_duplicate_day_download(self):
        files = CoinAPITradeFiles(SYMBOL)
        with (
            patch(MODULE + "find_trade_file", return_value="file") as find,
            patch(
                MODULE + "get_flat_file_content", return_value=archive([])
            ) as download,
            patch(MODULE + "pd.read_csv", wraps=pd.read_csv) as parse,
        ):
            for hour in (23, 22):
                start = datetime(2026, 6, 8, hour, tzinfo=UTC)
                self.assertEqual(
                    files.get_trades(start, start + timedelta(hours=1)), []
                )
            find.assert_called_once_with(SYMBOL, "20260608")
            download.assert_called_once()
            parse.assert_called_once()
            start = datetime(2026, 6, 9, tzinfo=UTC)
            files.get_trades(start, start + timedelta(hours=1))
            self.assertEqual(find.call_args.args, (SYMBOL, "2026060900"))

    def test_listing_pagination_exact_symbol_and_malformed_response(self):
        other = KEY.replace("_BTC_USDC+", "_BTC_USDC_OTHER+")
        with patch(
            MODULE + "get_flat_file_content",
            side_effect=[listing((other,), truncated="true", token="next"), listing()],
        ) as request:
            self.assertEqual(find_trade_file(SYMBOL, "2026111312"), KEY)
            self.assertEqual(
                request.call_args.kwargs["params"]["continuation-token"], "next"
            )
        for payload in (b"<Error/>", listing(truncated="true"), listing((KEY, KEY))):
            with (
                self.subTest(payload=payload),
                patch(MODULE + "get_flat_file_content", return_value=payload),
                self.assertRaises(ValueError),
            ):
                find_trade_file(SYMBOL, "2026111312")

    def test_transient_errors_retry_and_bad_rows_never_trigger_rest(self):
        with (
            patch(
                MODULE + "httpx2.get",
                side_effect=[response(b"busy", 503), response(listing())],
            ) as request,
            patch(MODULE + "time.sleep"),
        ):
            self.assertEqual(get_flat_file_content(), listing())
            self.assertEqual(request.call_count, 2)
        for rows in (
            [trade(), trade()],
            [trade(taker_side="UNKNOWN")],
            [trade(base_amount="NaN")],
        ):
            controller = CoinAPITrades(
                SimpleNamespace(api_symbol=SYMBOL), START, START, Mock()
            )
            candles = pd.DataFrame({"notional": [Decimal(1)]}, index=[START])
            with (
                self.subTest(rows=rows),
                patch(
                    MODULE + "httpx2.get",
                    side_effect=[response(listing()), response(archive(rows))],
                ),
                patch.object(controller, "fetch_rest_data_frame") as rest,
                self.assertRaises(ValueError),
            ):
                controller.fetch_trade_data_frame(
                    START, START + timedelta(minutes=1), candles
                )
            rest.assert_not_called()
