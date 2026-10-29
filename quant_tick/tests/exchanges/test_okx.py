from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import api, candles_api, trades_api
from quant_tick.exchanges.okx.api import contract_value
from quant_tick.exchanges.okx.candles import okx_candles
from quant_tick.exchanges.okx.controllers import OKXArchive, OKXTrades
from quant_tick.models import TradeData

from ..base import BaseSymbolTest

START = datetime(2025, 1, 1, tzinfo=UTC)
MS = int(START.timestamp() * 1000)


def trade(trade_id, timestamp=MS, market="BTC-USDT", size="1", price="100"):
    return {
        "instId": market,
        "tradeId": str(trade_id),
        "ts": str(timestamp),
        "sz": size,
        "px": price,
        "side": "buy",
    }


class OKXMarketTest(SimpleTestCase):
    def test_trade_cursor_preserves_ties_across_pages_and_half_open_bounds(self):
        controller = OKXTrades(
            SimpleNamespace(api_symbol="BTC-USDT"),
            START,
            START + timedelta(minutes=1),
            Mock(),
        )
        pages = [
            [trade(5, MS + 60000), trade(4), trade(3)],
            [trade(2), trade(1, MS - 1)],
        ]
        with patch(
            "quant_tick.exchanges.okx.controllers.get_okx_response", side_effect=pages
        ) as request:
            rows, _, _ = controller.iter_api(START, START + timedelta(minutes=1))
        frame = controller.get_data_frame(controller.parse_data(rows))
        self.assertEqual(frame.uid.tolist(), ["2", "3", "4"])
        self.assertEqual(request.call_args_list[0].args[1]["type"], "2")
        self.assertEqual(request.call_args_list[1].args[1]["type"], "1")
        self.assertEqual(request.call_args_list[1].args[1]["after"], "3")
        with (
            patch(
                "quant_tick.exchanges.okx.controllers.get_okx_response",
                side_effect=[pages[0], pages[0]],
            ),
            self.assertRaisesRegex(ValueError, "ordered"),
        ):
            controller.iter_api(START, START + timedelta(minutes=1))

    def test_linear_contract_size_uses_metadata_and_rejects_inverse(self):
        instrument = {
            "instId": "BTC-USDT-SWAP",
            "ctType": "linear",
            "ctValCcy": "BTC",
            "ctVal": "0.01",
        }
        controller = OKXTrades(
            SimpleNamespace(api_symbol="BTC-USDT-SWAP"),
            START,
            START + timedelta(minutes=1),
            Mock(),
        )
        with patch(
            "quant_tick.exchanges.okx.api.get_okx_response", return_value=[instrument]
        ) as request:
            rows = controller.parse_data(
                [
                    trade(2, market="BTC-USDT-SWAP", size="3"),
                    trade(1, market="BTC-USDT-SWAP", size="2"),
                ]
            )
        self.assertEqual(
            [row["notional"] for row in rows], [Decimal("0.03"), Decimal("0.02")]
        )
        self.assertEqual(rows[0]["volume"], Decimal(3))
        request.assert_called_once()
        with (
            patch(
                "quant_tick.exchanges.okx.api.get_okx_response",
                return_value=[{**instrument, "ctType": "inverse"}],
            ),
            self.assertRaisesRegex(ValueError, "linear"),
        ):
            contract_value("BTC-USDT-SWAP")

    def test_eight_hour_candles_use_completed_base_quantity_bars(self):
        pages = [
            [
                [
                    str(MS + 8 * 3600000),
                    "101",
                    "200",
                    "100",
                    "150",
                    "1000",
                    "10",
                    "1500",
                    "0",
                ],
                [
                    str(MS + 4 * 3600000),
                    "101",
                    "110",
                    "100",
                    "105",
                    "300",
                    "3",
                    "315",
                    "1",
                ],
                [str(MS), "100", "102", "99", "101", "200", "2", "202", "1"],
            ]
        ]
        with patch(
            "quant_tick.exchanges.okx.candles.get_okx_response", side_effect=pages
        ) as request:
            frame = okx_candles(
                "BTC-USDT-SWAP", START, START + timedelta(hours=9), resolution="8h"
            )
        self.assertEqual(request.call_args.args[1]["bar"], "4H")
        self.assertEqual(list(frame.index), [START])
        self.assertEqual(frame.iloc[0].notional, Decimal(5))
        self.assertEqual(frame.iloc[0].volume, Decimal(517))
        self.assertEqual(frame.iloc[0].close, Decimal(105))


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class OKXArchiveCollectionTest(BaseSymbolTest, TestCase):
    def test_archive_sides_and_known_overlap_preserve_utc_partitions(self):
        start = datetime(2023, 2, 9, tzinfo=UTC)
        ms = int(start.timestamp() * 1000)
        boundary = ms + 16 * 3600000
        symbol = self.get_symbol(exchange=Exchange.OKX, api_symbol="BTC-USDT")
        times = [
            ms - 1,
            ms,
            ms,
            boundary - 1435,
            *[boundary - 1341] * 4,
            boundary,
            ms + 24 * 3600000,
        ]
        sides = ("buy", "SELL", "BUY", "sell")
        rows = [
            [
                "BTC-USDT",
                str(397123642 + i),
                sides[i % len(sides)],
                str(100 + i),
                "1",
                str(ts),
            ]
            for i, ts in enumerate(times)
        ]
        readers = [
            ChunkReader(
                pd.DataFrame(
                    [list(OKXArchive.csv_columns), *rows[:5]],
                    columns=OKXArchive.csv_columns,
                ),
                pd.DataFrame(rows[5:8], columns=OKXArchive.csv_columns),
            ),
            ChunkReader(
                pd.DataFrame(
                    [list(OKXArchive.csv_columns), *rows[3:5]],
                    columns=OKXArchive.csv_columns,
                ),
                pd.DataFrame(rows[5:], columns=OKXArchive.csv_columns),
            ),
        ]
        candles = pd.DataFrame(
            {"notional": [Decimal(0)] * 1440},
            index=pd.date_range(start, periods=1440, freq="min"),
        )
        candles.index.name = "timestamp"
        candles.loc[start, "notional"] = Decimal(2)
        candles.loc[start + timedelta(hours=15, minutes=59), "notional"] = Decimal(5)
        candles.loc[start + timedelta(hours=16), "notional"] = Decimal(1)
        with (
            patch(
                "quant_tick.exchanges.okx.controllers.zip_chunk_downloader",
                side_effect=readers,
            ) as download,
            patch(
                "quant_tick.exchanges.okx.controllers.okx_candles", return_value=candles
            ),
        ):
            api(symbol, start, start + timedelta(days=1))
        self.assertIn("20230209/", download.call_args_list[0].args[0])
        self.assertIn("20230210/", download.call_args_list[1].args[0])
        self.assertTrue(all(reader.closed for reader in readers))
        partitions = TradeData.objects.filter(symbol=symbol)
        self.assertEqual(partitions.count(), 24)
        self.assertEqual(list(partitions.values_list("ok", flat=True)), [True] * 24)
        first = partitions.get(timestamp=start)
        self.assertEqual(first.uid, "397123643")
        first_trades = first.get_data_frame("raw_data")
        self.assertEqual(first_trades.price.tolist(), [Decimal(101), Decimal(102)])
        self.assertEqual(first_trades.tickRule.tolist(), [-1, 1])
        self.assertEqual(first_trades.notional.tolist(), [Decimal(1), Decimal(1)])
        self.assertEqual(first_trades.volume.tolist(), [Decimal(101), Decimal(102)])
        overlap_hour = partitions.get(timestamp=start + timedelta(hours=15))
        self.assertEqual(overlap_hour.uid, "397123645")
        overlap_trades = overlap_hour.get_data_frame("raw_data")
        self.assertEqual(
            overlap_trades.price.tolist(),
            [Decimal(value) for value in range(103, 108)],
        )
        self.assertEqual(overlap_trades.tickRule.tolist(), [-1, 1, -1, 1, -1])
        self.assertEqual(
            partitions.get(timestamp=start + timedelta(hours=16)).uid, "397123650"
        )

    def test_known_misfiled_trades_restore_and_retry_incomplete_hour(self):
        start = datetime(2022, 9, 14, tzinfo=UTC)
        symbol = self.get_symbol(exchange=Exchange.OKX, api_symbol="BTC-USDT")
        # Public September 14/15 files: two fills are missing from the first file.
        values = (
            ("372030792", "20370.5", "0.00261028", "1663123237478"),
            ("372030793", "20370.5", "0.00046621", "1663123237478"),
            ("372030794", "20370.5", "0.02953379", "1663123239265"),
            ("372030795", "20370.5", "0.00525621", "1663123239265"),
            ("372030796", "20371.3", "0.05", "1663123243647"),
            ("372158740", "20176.9", "0.00016852", "1663171196700"),
            ("372158741", "20176.9", "0.04956162", "1663171197713"),
            ("372158742", "20176.9", "0.04911295", "1663171200757"),
        )
        rows = [
            ["BTC-USDT", uid, "BUY", price, size, timestamp]
            for uid, price, size, timestamp in values
        ]
        columns = OKXArchive.csv_columns
        candles = pd.DataFrame(
            {"notional": [Decimal(0)] * 1440},
            index=pd.date_range(start, periods=1440, freq="min"),
        )
        candles.index.name = "timestamp"
        candles.loc[start + timedelta(hours=2, minutes=40), "notional"] = Decimal(
            "0.08786649"
        )
        candles.loc[start + timedelta(hours=15, minutes=59), "notional"] = Decimal(
            "0.04973014"
        )
        candles.loc[start + timedelta(hours=16), "notional"] = Decimal("0.04911295")
        hour = start + timedelta(hours=2)
        controller = OKXArchive(symbol, start, start + timedelta(days=1), Mock())
        for chunksize in (1, 3):
            with self.subTest(chunksize=chunksize):
                TradeData.objects.filter(symbol=symbol).delete()
                incomplete = controller.parse_dtypes_and_strip_columns(
                    pd.DataFrame([rows[0], rows[1], rows[4]], columns=columns)
                )
                TradeData.write(
                    symbol,
                    hour,
                    hour + timedelta(hours=1),
                    candles,
                    raw_trades=incomplete,
                )
                self.assertIs(TradeData.objects.get(symbol=symbol).ok, False)
                frames = [
                    pd.DataFrame(
                        [list(columns), rows[0], rows[1], rows[4], rows[5]],
                        columns=columns,
                    ),
                    pd.DataFrame(
                        [list(columns), *rows[1:5], *rows[6:]], columns=columns
                    ),
                ]
                readers = [
                    ChunkReader(
                        *(
                            frame.iloc[i : i + chunksize]
                            for i in range(0, len(frame), chunksize)
                        )
                    )
                    for frame in frames
                ]
                with (
                    patch(
                        "quant_tick.exchanges.okx.controllers.zip_chunk_downloader",
                        side_effect=readers,
                    ),
                    patch(
                        "quant_tick.exchanges.okx.controllers.okx_candles",
                        return_value=candles,
                    ),
                ):
                    api(symbol, start, start + timedelta(days=1), retry=True)
                self.assertTrue(all(reader.closed for reader in readers))
                partitions = TradeData.objects.filter(symbol=symbol)
                self.assertEqual(partitions.count(), 24)
                self.assertEqual(
                    list(partitions.values_list("ok", flat=True)), [True] * 24
                )
                repaired = partitions.get(timestamp=hour)
                self.assertEqual(repaired.uid, "372030792")
                stored = repaired.get_data_frame("raw_data")
                self.assertEqual(
                    stored.price.tolist(),
                    [Decimal("20370.5")] * 4 + [Decimal("20371.3")],
                )
                self.assertEqual(
                    stored.notional.tolist(), [Decimal(v[2]) for v in values[:5]]
                )
                self.assertEqual(
                    stored.volume.tolist(),
                    [Decimal(v[1]) * Decimal(v[2]) for v in values[:5]],
                )
                self.assertEqual(stored.tickRule.tolist(), [1] * 5)
                self.assertEqual(
                    stored.timestamp.tolist(),
                    pd.to_datetime(
                        [int(v[3]) for v in values[:5]], unit="ms", utc=True
                    ).tolist(),
                )
                before_boundary = partitions.get(timestamp=start + timedelta(hours=15))
                self.assertEqual(
                    before_boundary.get_data_frame("raw_data").notional.sum(),
                    Decimal("0.04973014"),
                )
                self.assertEqual(
                    partitions.get(timestamp=start + timedelta(hours=16)).uid,
                    "372158742",
                )

    def test_known_july_regressions_preserve_trades_and_quantity_failures(self):
        start = datetime(2022, 7, 21, tzinfo=UTC)
        ms = int(start.timestamp() * 1000)
        symbol = self.get_symbol(exchange=Exchange.OKX, api_symbol="BTC-USDT")
        columns = OKXArchive.csv_columns
        values = [
            ("358584060", "100", "1", ms + 16 * 3600000 - 1),
            ("358584473", "22683.0", "1.14515773", 1658419282000),
            ("358584474", "22683.0", "0.331715", 1658419282000),
            ("358584477", "22683.9", "0.00010359", 1658419281000),
            ("358584478", "22684.0", "0.00001332", 1658419281000),
            ("358700000", "101", "2", ms + 24 * 3600000 + 2000),
            ("358700001", "102", "3", ms + 24 * 3600000 + 1000),
            ("358912380", "103", "4", ms + 40 * 3600000 - 1000),
            ("358912381", "104", "5", ms + 40 * 3600000 + 100),
        ]
        rows = [
            ["BTC-USDT", uid, "BUY", price, size, str(timestamp)]
            for uid, price, size, timestamp in values
        ]
        archives = {
            "2022-07-21": [rows[0]],
            "2022-07-22": rows[1:8],
            "2022-07-23": [rows[8]],
        }
        candles = pd.DataFrame(
            {"notional": [Decimal(0)] * 2880},
            index=pd.date_range(start, periods=2880, freq="min", name="timestamp"),
        )
        for _, _, size, timestamp in values:
            minute = pd.Timestamp(timestamp, unit="ms", tz="UTC").floor("min")
            candles.loc[minute, "notional"] += Decimal(size)
        # The public file also lacks quantity; ordering must not hide that result.
        missing_hour = start + timedelta(days=1)
        candles.loc[missing_hour, "notional"] = Decimal(6)
        for chunksize in (1, 3):
            with self.subTest(chunksize=chunksize):
                TradeData.objects.filter(symbol=symbol).delete()
                readers = []

                def download(url, *args, readers=readers, **kwargs):
                    day = url.rsplit("-trades-", 1)[1].removesuffix(".zip")
                    size = kwargs["chunksize"]
                    frame = pd.DataFrame(
                        [list(columns), *archives[day]], columns=columns
                    )
                    reader = ChunkReader(
                        *(frame.iloc[i : i + size] for i in range(0, len(frame), size))
                    )
                    readers.append(reader)
                    return reader

                with (
                    patch.object(OKXArchive, "archive_chunksize", chunksize),
                    patch(
                        "quant_tick.exchanges.okx.controllers.zip_chunk_downloader",
                        side_effect=download,
                    ),
                    patch(
                        "quant_tick.exchanges.okx.controllers.okx_candles",
                        return_value=candles,
                    ),
                ):
                    api(symbol, start, start + timedelta(days=2))
                self.assertEqual(len(readers), 4)
                self.assertTrue(all(reader.closed for reader in readers))
                partitions = TradeData.objects.filter(symbol=symbol)
                self.assertEqual(partitions.count(), 48)
                self.assertEqual(
                    list(
                        partitions.filter(ok=False).values_list("timestamp", flat=True)
                    ),
                    [missing_hour],
                )
                self.assertEqual(partitions.filter(ok=True).count(), 47)
                stored = pd.concat(
                    [
                        row.get_data_frame("raw_data")
                        for row in partitions
                        if row.raw_data
                    ],
                    ignore_index=True,
                )
                expected = sorted(values, key=lambda row: row[3])
                self.assertEqual(
                    stored.timestamp.tolist(),
                    pd.to_datetime(
                        [row[3] for row in expected], unit="ms", utc=True
                    ).tolist(),
                )
                self.assertEqual(
                    stored.price.tolist(), [Decimal(row[1]) for row in expected]
                )
                self.assertEqual(
                    stored.notional.tolist(), [Decimal(row[2]) for row in expected]
                )
                self.assertEqual(
                    stored.volume.tolist(),
                    [Decimal(row[1]) * Decimal(row[2]) for row in expected],
                )
                self.assertEqual(stored.tickRule.tolist(), [1] * len(values))
                self.assertEqual(
                    partitions.get(timestamp=start + timedelta(hours=16)).uid,
                    "358584477",
                )
                self.assertEqual(
                    partitions.get(timestamp=missing_hour).uid, "358700001"
                )
                self.assertEqual(
                    partitions.get(timestamp=start + timedelta(hours=40)).uid,
                    "358912381",
                )

    def test_missing_second_archive_does_not_write_partial_day(self):
        symbol = self.get_symbol(exchange=Exchange.OKX, api_symbol="BTC-USDT")
        reader = ChunkReader(pd.DataFrame(columns=OKXArchive.csv_columns))
        with patch(
            "quant_tick.exchanges.okx.controllers.zip_chunk_downloader",
            side_effect=[reader, None],
        ):
            api(symbol, START, START + timedelta(days=1))
        self.assertTrue(reader.closed)
        self.assertFalse(TradeData.objects.filter(symbol=symbol).exists())

    def test_dispatch_rejects_mismatched_market_types(self):
        for market, symbol_type in (
            ("BTC-USDT", SymbolType.PERPETUAL),
            ("BTC-USDT-SWAP", SymbolType.SPOT),
        ):
            symbol = self.get_symbol(
                exchange=Exchange.OKX, api_symbol=market, symbol_type=symbol_type
            )
            with self.assertRaises(ValueError):
                trades_api(symbol, START, START + timedelta(minutes=1), Mock())
            with self.assertRaises(ValueError):
                candles_api(symbol, START, START + timedelta(minutes=1))

    def test_candle_request_failure_closes_unconsumed_archives(self):
        symbol = self.get_symbol(exchange=Exchange.OKX, api_symbol="BTC-USDT")
        readers = [
            ChunkReader(pd.DataFrame(columns=OKXArchive.csv_columns)) for _ in range(2)
        ]
        with (
            patch(
                "quant_tick.exchanges.okx.controllers.zip_chunk_downloader",
                side_effect=readers,
            ),
            patch(
                "quant_tick.exchanges.okx.controllers.okx_candles",
                side_effect=RuntimeError("venue unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "venue unavailable"),
        ):
            api(symbol, START, START + timedelta(days=1))
        self.assertTrue(all(reader.closed for reader in readers))
        self.assertFalse(TradeData.objects.filter(symbol=symbol).exists())


class ChunkReader:
    def __init__(self, *frames):
        self.frames = iter(frames)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.frames)

    def close(self):
        self.closed = True
