import gzip
import json
import os
import re
import sqlite3
from contextlib import nullcontext
from copy import deepcopy
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import grpc
import pandas as pd
from django.core.files.storage import default_storage
from django.core.management import call_command
from django.db import transaction
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import CheckpointType, Exchange, Frequency, SymbolType
from quant_tick.exchanges.order_book.book import Book
from quant_tick.exchanges.order_book.checkpoints import encode, load, persist
from quant_tick.exchanges.order_book.collector import order_book
from quant_tick.exchanges.order_book.replay import replay
from quant_tick.exchanges.order_book.storage import (
    compact_partition,
    convert_order_book_data_to_daily,
    iter_rows,
)
from quant_tick.exchanges.quicknode import book_history, book_pb2, book_snapshot
from quant_tick.models import Checkpoint, OrderBookData, Symbol
from quant_tick.services.task_lease import TaskLeaseLost


class BookFixture:
    def setUp(self):
        super().setUp()
        self.start = pd.Timestamp("2026-11-20T23:59Z")
        self.end = self.start + pd.Timedelta(minutes=2)
        self.seed = {
            "version": 1,
            "coin": "BTC",
            "block_number": 100,
            "timestamp": self.start.isoformat(),
            "boundary": True,
            "orders": [
                [i + 1, "B", str(100 - i), "10" if i == 0 else "1"] for i in range(25)
            ]
            + [
                [i + 101, "A", str(101 + i), "12" if i == 0 else "1"] for i in range(25)
            ],
        }
        self.groups = []
        self.group(
            101,
            10,
            [["B", "100", 1, 0, "update", None, "10", "8"]],
            [("A", "100", "2")],
        )
        self.group(102, 20, [["B", "100", 999, 0, "new", "5", None, None]])
        self.group(103, 30, [["B", "99", 2, 0, "remove", None, None, None]])
        self.group(
            105,
            60,
            [["A", "101", 101, 0, "update", None, "12", "9"]],
            [("B", "101", "3")],
        )
        self.group(110, 125, [["B", "100", 1000, 0, "new", "0.3", None, None]])
        book = Book(self.seed, "BTC")
        for _, group in self.groups:
            for row in group["updates"]:
                book.apply(row)
        self.terminal = book.state(self.start + pd.Timedelta(seconds=130))

    def group(self, height, seconds, updates, trades=()):
        moment = self.start + pd.Timedelta(seconds=seconds)
        self.groups.append(
            (
                height,
                {
                    "time": moment,
                    "updates": [[height, moment, *row] for row in updates],
                    "trades": [
                        {"side": side, "price": price, "size": size}
                        for side, price, size in trades
                    ],
                },
            )
        )

    def history(self, coin, start, terminal, **kwargs):
        return [
            (height, group)
            for height, group in self.groups
            if start["block_number"] < height <= terminal["block_number"]
        ]


class OrderBookReplayTest(BookFixture, SimpleTestCase):
    def test_forward_replay_measurements_and_checkpoint_resume(self):
        rows, states = replay(
            self.seed, self.terminal, self.groups, self.start, self.end
        )
        self.assertEqual(
            [moment for moment, _ in rows],
            [self.start, self.start + pd.Timedelta(minutes=1)],
        )
        first, second = [values for _, values in rows]
        self.assertEqual(first["coverage_us"], 60_000_000)
        self.assertEqual(first["sell_notional"], 2)
        self.assertEqual(first["sell_volume"], 200)
        self.assertEqual(first["bid_net_addition_notional_5"], 4)
        self.assertEqual(first["bid_executed_notional_5"], 2)
        self.assertEqual(second["buy_notional"], 3)
        self.assertEqual(second["ask_net_addition_notional_5"], 0)
        self.assertEqual(second["ask_executed_notional_5"], 3)
        self.assertEqual(second["bid_net_addition_notional_5"], 0)
        midpoint = self.start + pd.Timedelta(minutes=1)
        restored = json.loads(gzip.decompress(states[midpoint][1]))
        resumed, later_states = replay(
            restored,
            self.terminal,
            self.history("BTC", restored, self.terminal),
            midpoint,
            self.end,
        )
        self.assertEqual(resumed, rows[1:])
        self.assertEqual(later_states[self.end], states[self.end])
        boundary = json.loads(gzip.decompress(states[self.end][1]))
        self.assertNotIn(1000, {row[0]: row for row in boundary["orders"]})

    def test_partial_start_and_quiet_minutes_do_not_invent_coverage(self):
        seed = deepcopy(self.seed)
        seed["timestamp"] = (self.start + pd.Timedelta(seconds=25)).isoformat()
        terminal = deepcopy(seed)
        terminal["timestamp"] = (self.end + pd.Timedelta(seconds=5)).isoformat()
        rows, states = replay(seed, terminal, [], self.start, self.end)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], self.start + pd.Timedelta(minutes=1))
        self.assertEqual(rows[0][1]["coverage_us"], 60_000_000)
        self.assertEqual(rows[0][1]["trade_count"], 0)
        self.assertEqual(rows[0][1]["bid_notional_5_mean"], 14)
        self.assertIn(self.end, states)

        # A live snapshot includes its block, even exactly on a minute boundary.
        seed["timestamp"] = self.start.isoformat()
        seed["boundary"] = False
        live_rows, _ = replay(seed, terminal, [], self.start, self.end)
        self.assertEqual(live_rows, rows)

    def test_nanosecond_updates_preserve_complete_time_weighting(self):
        groups = deepcopy(self.groups)
        for index, (_, group) in enumerate(groups):
            delta = pd.Timedelta(nanoseconds=137 * (index + 1))
            group["time"] += delta
            for row in group["updates"]:
                row[1] += delta
        rows, _ = replay(self.seed, self.terminal, groups, self.start, self.end)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(values["coverage_us"] == 60_000_000 for _, values in rows))

    def test_missing_inconsistent_and_reversed_history_is_rejected(self):
        for problem in ("terminal", "remove", "original_size", "order"):
            groups = deepcopy(self.groups)
            terminal = deepcopy(self.terminal)
            if problem == "terminal":
                terminal["orders"][0][3] = "99"
            elif problem == "remove":
                groups[2][1]["updates"][0][4] = 123456789
            elif problem == "original_size":
                groups[0][1]["updates"][0][8] = "99"
            else:
                groups.reverse()
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                replay(self.seed, terminal, groups, self.start, self.end)


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class OrderBookCollectionTest(BookFixture, TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = Symbol.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            save_order_book=True,
        )
        self.env = patch.dict(os.environ, {"QUICKNODE_API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def initialize(self):
        with (
            patch(
                "quant_tick.exchanges.order_book.collector.get_current_time",
                return_value=self.start,
            ),
            patch(
                "quant_tick.exchanges.order_book.collector.snapshot",
                return_value=self.seed,
            ),
        ):
            call_command("order_book", code_name=[self.symbol.code_name])

    def test_bootstrap_forward_collection_retry_and_noop(self):
        self.initialize()
        self.assertEqual(OrderBookData.objects.count(), 0)
        seed = Checkpoint.objects.get()
        self.assertEqual(seed.checkpoint_type, CheckpointType.HYPERLIQUID_ORDER_BOOK)
        self.assertIsInstance(seed.json_data["orders"], int)
        self.assertEqual(seed.json_data["orders"], 50)
        self.assertTrue(default_storage.exists(seed.file_data.name))
        with (
            patch(
                "quant_tick.exchanges.order_book.collector.get_current_time",
                return_value=self.end + pd.Timedelta(minutes=1),
            ),
            patch(
                "quant_tick.exchanges.order_book.collector.snapshot",
                return_value=self.terminal,
            ) as snapshot,
            patch(
                "quant_tick.exchanges.order_book.collector.history",
                side_effect=self.history,
            ) as history,
        ):
            order_book(self.symbol, self.start, self.end)
            original = list(iter_rows(self.symbol))
            self.assertEqual(len(original), 2)
            order_book(self.symbol, self.start, self.end)
            self.assertEqual(snapshot.call_count, 1)
            self.assertEqual(history.call_count, 1)
            order_book(self.symbol, self.start, self.end, retry=True)
            self.assertEqual(list(iter_rows(self.symbol)), original)
            self.assertEqual(snapshot.call_count, 1)
        self.assertEqual(Checkpoint.objects.count(), 3)

    def test_terminal_mismatch_preserves_previous_snapshot_and_rows(self):
        self.initialize()
        original = Checkpoint.objects.get()
        bad = deepcopy(self.terminal)
        bad["orders"][0][3] = "999"
        with (
            patch(
                "quant_tick.exchanges.order_book.collector.get_current_time",
                return_value=self.end,
            ),
            patch(
                "quant_tick.exchanges.order_book.collector.snapshot", return_value=bad
            ),
            patch(
                "quant_tick.exchanges.order_book.collector.history",
                side_effect=self.history,
            ),
            self.assertRaisesRegex(ValueError, "does not reconcile"),
        ):
            order_book(self.symbol, self.start, self.end)
        self.assertEqual(Checkpoint.objects.count(), 1)
        self.assertEqual(OrderBookData.objects.count(), 0)
        self.assertEqual(load(original)["block_number"], 100)

    def test_lagging_round_trip_updates_are_read_before_advancing_checkpoint(self):
        self.seed["orders"] = [self.seed["orders"][0], self.seed["orders"][25]]
        self.initialize()
        terminal = deepcopy(self.seed)
        terminal.update(block_number=120, timestamp=self.end.isoformat())
        events = [
            [
                101,
                self.start + pd.Timedelta(seconds=10),
                "B",
                "100",
                999,
                0,
                "new",
                "8",
                None,
                None,
            ],
            [
                102,
                self.start + pd.Timedelta(seconds=40),
                "B",
                "100",
                999,
                0,
                "remove",
                None,
                None,
                None,
            ],
        ]
        polls = 0

        def query(sql):
            nonlocal polls
            if "max(block_number)" in sql:
                polls += 1
                height = 100 if polls == 1 else 120
                return [{"height": height}, {"height": height}]
            if "groupArray" in sql:
                first_window = book_history.time_literal(self.start) in sql
                return [{"events": events if polls >= 2 and first_window else []}]
            return []

        with (
            patch(
                "quant_tick.exchanges.order_book.collector.get_current_time",
                return_value=self.end,
            ),
            patch(
                "quant_tick.exchanges.order_book.collector.snapshot",
                return_value=terminal,
            ),
            patch.object(book_history, "query", side_effect=query),
            patch.object(book_history.time, "sleep"),
        ):
            order_book(self.symbol, self.start, self.end)
        first = OrderBookData.objects.get(timestamp=self.start)
        values = first.get_data_frame().iloc[0]
        self.assertEqual(values["bid_notional_5_mean"], 14)
        self.assertEqual(values["book_update_count"], 2)
        checkpoint = Checkpoint.objects.get(timestamp=self.end)
        self.assertEqual(load(checkpoint)["orders"], self.seed["orders"])

    def test_upload_database_and_lease_failures_do_not_delete_previous_data(self):
        self.initialize()
        checkpoint = Checkpoint.objects.get()
        rows, states = replay(
            self.seed, self.terminal, self.groups, self.start, self.end
        )
        uploaded = []
        save = default_storage.save

        def fail_after_first_minute(name, content, **kwargs):
            if len(uploaded) == len(states) + 1:
                raise RuntimeError("upload failed")
            saved = save(name, content, **kwargs)
            uploaded.append(saved)
            return saved

        for target in ("storage", "database", "lease"):
            patches = {
                "storage": patch.object(
                    default_storage, "save", side_effect=fail_after_first_minute
                ),
                "database": patch.object(
                    OrderBookData.objects,
                    "bulk_create",
                    side_effect=RuntimeError("database failed"),
                ),
                "lease": nullcontext(),
            }
            owned = (
                MagicMock(
                    side_effect=[None] * len(states) + [TaskLeaseLost("lease lost")]
                )
                if target == "lease"
                else None
            )
            with (
                self.subTest(target=target),
                patches[target],
                self.assertRaises(RuntimeError),
            ):
                persist(self.symbol, rows, states, assert_lease_owned=owned)
            self.assertEqual(Checkpoint.objects.count(), 1)
            self.assertEqual(OrderBookData.objects.count(), 0)
            self.assertEqual(load(checkpoint)["orders"], self.seed["orders"])
            self.assertTrue(all(not default_storage.exists(name) for name in uploaded))

    def test_checkpoint_deletion_only_removes_file_after_commit(self):
        self.initialize()
        checkpoint = Checkpoint.objects.get()
        name = checkpoint.file_data.name
        with self.assertRaises(RuntimeError), transaction.atomic():
            checkpoint.delete()
            raise RuntimeError("rollback")
        self.assertTrue(default_storage.exists(name))
        with self.captureOnCommitCallbacks(execute=True):
            Checkpoint.objects.only("pk").delete()
        self.assertFalse(default_storage.exists(name))

    def test_retention_keeps_first_hourly_and_latest_resume_states(self):
        self.initialize()
        original = Checkpoint.objects.get()
        names = {}
        for offset in range(1, 5):
            boundary = self.start + pd.Timedelta(minutes=offset)
            state = Book(self.seed, "BTC").state(boundary)
            with self.captureOnCommitCallbacks(execute=True):
                persist(self.symbol, [], {boundary: encode(state)})
            names[boundary] = Checkpoint.objects.get(timestamp=boundary).file_data.name
        expected = [self.start + pd.Timedelta(minutes=i) for i in (0, 1, 3, 4)]
        self.assertEqual(
            list(
                Checkpoint.objects.order_by("timestamp").values_list(
                    "timestamp", flat=True
                )
            ),
            expected,
        )
        self.assertTrue(default_storage.exists(original.file_data.name))
        for boundary, name in names.items():
            self.assertEqual(default_storage.exists(name), boundary in expected)

    def test_bad_checksum_and_earlier_history_are_not_silently_reseeded(self):
        self.initialize()
        checkpoint = Checkpoint.objects.get()
        checkpoint.json_data["sha256"] = "bad"
        checkpoint.save(update_fields=["json_data"])
        with self.assertRaisesRegex(ValueError, "checksum"):
            load(checkpoint)
        with patch("quant_tick.exchanges.order_book.collector.snapshot") as snapshot:
            order_book(
                self.symbol,
                self.start - pd.Timedelta(days=1),
                self.start - pd.Timedelta(hours=1),
                retry=True,
            )
        snapshot.assert_not_called()

    def test_export_preserves_timestamps_and_numeric_features(self):
        rows, _ = replay(self.seed, self.terminal, self.groups, self.start, self.end)
        persist(self.symbol, rows, {})
        with TemporaryDirectory() as folder:
            output = Path(folder) / "book.parquet"
            call_command(
                "export_order_book",
                code_name=self.symbol.code_name,
                output=output,
                stdout=StringIO(),
            )
            frame = pd.read_parquet(output)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.sell_notional.tolist(), [2, 0])
        self.assertEqual(
            frame.timestamp.tolist(), [self.start, self.start + pd.Timedelta(minutes=1)]
        )


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class OrderBookStorageTest(BookFixture, TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = Symbol.objects.create(
            exchange=Exchange.HYPERLIQUID, api_symbol="BTC"
        )
        self.day = self.start.floor("D")
        rows, _ = replay(self.seed, self.terminal, self.groups, self.start, self.end)
        self.values = [values for _, values in rows]

    def populate(self, count):
        rows = [
            (
                self.day + pd.Timedelta(minutes=i),
                {**self.values[i % 2], "trade_count": Decimal(i)},
            )
            for i in range(count)
        ]
        persist(self.symbol, rows, {})
        return rows

    def export(self, **options):
        with TemporaryDirectory() as folder:
            output = Path(folder) / "book.parquet"
            call_command(
                "export_order_book",
                code_name=self.symbol.code_name,
                output=output,
                stdout=StringIO(),
                **options,
            )
            return pd.read_parquet(output)

    def test_hour_day_compaction_and_partial_exports_keep_minute_values(self):
        self.populate(1440)
        minute_files = list(OrderBookData.objects.values_list("file_data", flat=True))
        self.assertEqual(len(minute_files), 1440)
        self.assertTrue(
            all(name and default_storage.exists(name) for name in minute_files)
        )
        original = self.export()
        hour_end = self.day + pd.Timedelta(hours=1)
        with self.captureOnCommitCallbacks(execute=True):
            convert_order_book_data_to_daily(self.symbol, self.day, hour_end)
        first = OrderBookData.objects.get(timestamp=self.day)
        self.assertEqual(first.frequency, Frequency.HOUR)
        self.assertTrue(
            all(not default_storage.exists(name) for name in minute_files[:60])
        )
        self.assertTrue(default_storage.exists(minute_files[60]))
        self.assertEqual(len(first.get_data_frame()), 60)
        self.assertTrue(default_storage.exists(first.file_data.name))
        pd.testing.assert_frame_equal(
            self.export(
                date_from=str(self.day + pd.Timedelta(minutes=15)),
                date_to=str(self.day + pd.Timedelta(minutes=75)),
            ),
            original.iloc[15:75].reset_index(drop=True),
        )
        with (
            self.captureOnCommitCallbacks(execute=True),
            patch(
                "quant_tick.management.commands.convert_order_book_data_to_daily.get_current_time",
                return_value=self.day + pd.Timedelta(days=1, hours=4),
            ),
        ):
            call_command(
                "convert_order_book_data_to_daily",
                code_name=[self.symbol.code_name],
                date_from=str(self.day.date()),
                date_to=str((self.day + pd.Timedelta(days=1)).date()),
            )
        daily = OrderBookData.objects.get(symbol=self.symbol)
        self.assertEqual(daily.frequency, Frequency.DAY)
        self.assertEqual(len(daily.get_data_frame()), 1440)
        self.assertFalse(default_storage.exists(first.file_data.name))
        pd.testing.assert_frame_equal(self.export(), original)
        pd.testing.assert_frame_equal(
            self.export(
                date_from=str(self.day + pd.Timedelta(minutes=15)),
                date_to=str(self.day + pd.Timedelta(minutes=75)),
            ),
            original.iloc[15:75].reset_index(drop=True),
        )
        # Already compacted data is unchanged on the next scheduled pass.
        convert_order_book_data_to_daily(
            self.symbol, self.day, self.day + pd.Timedelta(days=1)
        )
        self.assertEqual(
            OrderBookData.objects.get().file_data.name, daily.file_data.name
        )

    def test_retry_replaces_files_and_preserves_other_minutes_in_each_partition(self):
        rows = self.populate(1440)
        minute = self.day + pd.Timedelta(minutes=15)
        for frequency in (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY):
            with self.subTest(frequency=frequency):
                end = self.day + pd.Timedelta(minutes=frequency)
                with self.captureOnCommitCallbacks(execute=True):
                    convert_order_book_data_to_daily(self.symbol, self.day, end)
                boundary = minute if frequency == Frequency.MINUTE else self.day
                old = OrderBookData.objects.get(timestamp=boundary)
                changed = {
                    **rows[15][1],
                    "buy_volume": Decimal("12345.678901") + frequency,
                }
                with self.captureOnCommitCallbacks(execute=True):
                    persist(self.symbol, [(minute, changed)], {})
                stored = OrderBookData.objects.get(timestamp=boundary)
                self.assertEqual(stored.frequency, frequency)
                self.assertNotEqual(stored.file_data.name, old.file_data.name)
                self.assertFalse(default_storage.exists(old.file_data.name))
                self.assertEqual(
                    OrderBookData.objects.filter(timestamp=minute).exists(),
                    frequency == Frequency.MINUTE,
                )
                result = list(iter_rows(self.symbol))
                self.assertEqual(len(result), 1440)
                self.assertEqual(result[15]["buy_volume"], changed["buy_volume"])
                self.assertEqual(
                    [row["trade_count"] for row in result], list(range(1440))
                )
                for index in (0, 14, 16, 59, 60, 1439):
                    self.assertEqual(
                        result[index]["buy_volume"], rows[index][1]["buy_volume"]
                    )

    def test_failed_compaction_and_retry_keep_files_rows_and_checkpoints(self):
        rows = self.populate(60)
        hour_end = self.day + pd.Timedelta(hours=1)
        owned = MagicMock(side_effect=[None, TaskLeaseLost("lost")])
        with (
            patch.object(default_storage, "save", wraps=default_storage.save) as saved,
            self.assertRaises(TaskLeaseLost),
        ):
            convert_order_book_data_to_daily(
                self.symbol, self.day, hour_end, assert_lease_owned=owned
            )
        self.assertEqual(OrderBookData.objects.count(), 60)
        self.assertEqual(Checkpoint.objects.count(), 0)
        self.assertEqual(saved.call_count, 1)
        self.assertFalse(default_storage.exists(saved.call_args.args[0]))
        convert_order_book_data_to_daily(self.symbol, self.day, hour_end)
        original = OrderBookData.objects.get()
        checkpoint_state = Book(self.seed, "BTC").state(self.start)
        retry_rows = [(rows[15][0], {**rows[15][1], "buy_volume": Decimal(999)})]
        for stage in ("upload", "database", "lease"):
            with self.subTest(stage=stage):
                owned = (
                    MagicMock(side_effect=[None, None, TaskLeaseLost("lost")])
                    if stage == "lease"
                    else None
                )
                failure = {
                    "upload": patch.object(
                        default_storage,
                        "save",
                        side_effect=RuntimeError("upload failed"),
                    ),
                    "database": patch.object(
                        OrderBookData.objects,
                        "bulk_create",
                        side_effect=RuntimeError("database failed"),
                    ),
                    "lease": nullcontext(),
                }[stage]
                with failure, self.assertRaises(RuntimeError):
                    persist(
                        self.symbol,
                        retry_rows,
                        {self.start: encode(checkpoint_state)},
                        assert_lease_owned=owned,
                    )
                self.assertEqual(Checkpoint.objects.count(), 0)
                self.assertEqual(
                    OrderBookData.objects.get().file_data.name, original.file_data.name
                )
                self.assertTrue(default_storage.exists(original.file_data.name))
                self.assertEqual(
                    list(iter_rows(self.symbol))[15]["buy_volume"],
                    rows[15][1]["buy_volume"],
                )
        with self.assertRaises(RuntimeError), transaction.atomic():
            OrderBookData.objects.all().delete()
            raise RuntimeError("rollback")
        self.assertTrue(default_storage.exists(original.file_data.name))
        with self.captureOnCommitCallbacks(execute=True):
            self.symbol.delete()
        self.assertFalse(default_storage.exists(original.file_data.name))

    def test_missing_minutes_leave_minute_files_and_late_retry_can_complete_hour(self):
        rows = self.populate(60)
        OrderBookData.objects.filter(timestamp=rows[15][0]).delete()
        end = self.day + pd.Timedelta(hours=1)
        convert_order_book_data_to_daily(self.symbol, self.day, end)
        self.assertEqual(OrderBookData.objects.count(), 59)
        self.assertFalse(OrderBookData.objects.filter(file_data="").exists())
        self.assertFalse(
            OrderBookData.objects.exclude(frequency=Frequency.MINUTE).exists()
        )
        persist(self.symbol, [rows[15]], {})
        # Candidate selection precedes the symbol lock. Read current rows again
        # after acquiring it, including any retry that finished in between.
        updated = {**rows[16][1], "buy_volume": Decimal(42)}
        owned = MagicMock()

        def update_before_read():
            if owned.call_count == 1:
                persist(self.symbol, [(rows[16][0], updated)], {})

        owned.side_effect = update_before_read
        compact_partition(
            self.symbol,
            self.day,
            Frequency.MINUTE,
            Frequency.HOUR,
            assert_lease_owned=owned,
        )
        self.assertEqual(OrderBookData.objects.get().frequency, Frequency.HOUR)
        self.assertEqual(list(iter_rows(self.symbol))[16]["buy_volume"], 42)


class QuickNodeBookAdapterTest(BookFixture, SimpleTestCase):
    def test_off_book_transfers_do_not_change_execution_or_replenishment_totals(self):
        database = sqlite3.connect(":memory:")
        self.addCleanup(database.close)
        database.row_factory = sqlite3.Row
        database.create_function("toString", 1, str)
        database.execute(
            "CREATE TABLE hyperliquid_trades (block_number INTEGER, block_time TEXT, "
            "coin TEXT, trade_id INTEGER, side TEXT, price TEXT, size TEXT, "
            "buyer_dir TEXT, seller_dir TEXT, liquidation_method TEXT)"
        )
        records = []
        for height, group in self.groups:
            for trade in group["trades"]:
                records.append(
                    (
                        height,
                        group["time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "BTC",
                        len(records),
                        trade["side"],
                        trade["price"],
                        trade["size"],
                        "Open Long",
                        "Open Short",
                        "market" if trade["side"] == "B" else None,
                    )
                )
        off_book = (
            ("Auto-Deleveraging", "Close Long", None),
            ("Close Short", "Auto-Deleveraging", "backstop"),
            ("Open Long", "Close Long", "backstop"),
            ("Net Child Vaults", "Net Child Vaults", None),
            ("Partial Borrow Liquidation", "Partial Borrow Liquidation", None),
            ("Backstop Borrow Liquidation", "Backstop Borrow Liquidation", None),
        )
        for buyer, seller, method in off_book:
            records.append(
                (
                    101,
                    records[0][1],
                    "BTC",
                    len(records),
                    "A",
                    "100",
                    "1000",
                    buyer,
                    seller,
                    method,
                )
            )
        database.executemany(
            "INSERT INTO hyperliquid_trades VALUES (?,?,?,?,?,?,?,?,?,?)", records
        )

        def query(sql):
            if "groupArray" in sql:
                lower = pd.Timestamp(
                    re.search(r"block_time >= '([^']+)'", sql)[1], tz="UTC"
                )
                upper = pd.Timestamp(
                    re.search(r"block_time < '([^']+)'", sql)[1], tz="UTC"
                )
                return [
                    {
                        "events": [
                            row
                            for _, group in self.groups
                            for row in group["updates"]
                            if lower <= group["time"] < upper
                        ]
                    }
                ]
            # Execute the production predicate; FINAL is ClickHouse-specific.
            return [dict(row) for row in database.execute(sql.replace(" FINAL ", " "))]

        expected = replay(self.seed, self.terminal, self.groups, self.start, self.end)
        with (
            patch.object(book_history, "query", side_effect=query),
            patch.object(book_history, "indexed_through"),
        ):
            actual = replay(
                self.seed,
                self.terminal,
                book_history.history("BTC", self.seed, self.terminal),
                self.start,
                self.end,
            )
        self.assertEqual(actual, expected)

    def test_sql_occurrence_order_and_index_lag(self):
        terminal = deepcopy(self.seed)
        terminal["block_number"] = 110
        with (
            patch.object(
                book_history,
                "query",
                side_effect=[
                    [{"height": 109}, {"height": 110}],
                    [{"height": 110}, {"height": 110}],
                ],
            ),
            patch.object(book_history.time, "sleep") as sleep,
        ):
            book_history.indexed_through(terminal, retry=1)
        sleep.assert_called_once_with(1)
        moment = self.start + pd.Timedelta(seconds=1)
        events = [
            [101, moment.isoformat(), "B", "100", 999, 0, "remove", None, None, None],
            [101, moment.isoformat(), "B", "100", 999, 1, "new", "1", None, None],
        ]
        terminal["timestamp"] = (self.start + pd.Timedelta(seconds=2)).isoformat()
        with (
            patch.object(book_history, "indexed_through"),
            patch.object(book_history, "query", side_effect=[[{"events": events}], []]),
        ):
            groups = list(book_history.history("BTC", self.seed, terminal))
        self.assertEqual([row[6] for row in groups[0][1]["updates"]], ["new", "remove"])

    def test_snapshot_retry_after_and_credential_redaction(self):
        update = book_pb2.L4BookUpdatesUpdate(
            time=int(self.start.timestamp() * 1000), height=100, snapshot=True
        )
        for oid, side, price, size in self.seed["orders"]:
            update.diffs.add(
                diff_type=1, coin="BTC", oid=oid, side=side, px=price, sz=size
            )

        class Failure(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.RESOURCE_EXHAUSTED

            def trailing_metadata(self):
                return (("retry-after", "3"),)

            def __str__(self):
                return "https://node.quiknode.pro/private-token"

        failed = MagicMock()
        failed.__next__.side_effect = Failure()
        succeeded = MagicMock()
        succeeded.__next__.return_value = update
        channel = MagicMock()
        channel.__enter__.return_value = channel
        channel.unary_stream.return_value.side_effect = [failed, succeeded, failed]
        with (
            patch.object(
                book_snapshot,
                "endpoint",
                return_value=("node.quiknode.pro:10000", "private-token"),
            ),
            patch.object(book_snapshot.grpc, "secure_channel", return_value=channel),
            patch.object(book_snapshot.time, "sleep") as sleep,
            self.assertLogs(book_snapshot.logger, "INFO") as logs,
        ):
            state = book_snapshot.snapshot("BTC", retry=1)
            with self.assertRaises(RuntimeError) as failure:
                book_snapshot.snapshot("BTC", retry=0)
        self.assertEqual(Book(state, "BTC").orders, Book(self.seed, "BTC").orders)
        sleep.assert_called_once_with(3)
        self.assertIsNone(failure.exception.__context__)
        self.assertNotIn(
            "private-token", str(failure.exception) + " ".join(logs.output)
        )
        self.assertEqual(failed.cancel.call_count, 2)
        succeeded.cancel.assert_called_once()
