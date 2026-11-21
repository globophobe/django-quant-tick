import gzip
import json
import os
from contextlib import nullcontext
from copy import deepcopy
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

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.order_book.book import Book
from quant_tick.exchanges.order_book.checkpoints import encode, load, persist
from quant_tick.exchanges.order_book.collector import order_book
from quant_tick.exchanges.order_book.replay import replay
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
            collect_order_book=True,
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
        self.assertIsInstance(seed.state["orders"], int)
        self.assertEqual(seed.state["orders"], 50)
        self.assertTrue(default_storage.exists(seed.data.name))
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
            original = list(OrderBookData.objects.values("timestamp", "json_data"))
            self.assertEqual(len(original), 2)
            order_book(self.symbol, self.start, self.end)
            self.assertEqual(snapshot.call_count, 1)
            self.assertEqual(history.call_count, 1)
            order_book(self.symbol, self.start, self.end, retry=True)
            self.assertEqual(
                list(OrderBookData.objects.values("timestamp", "json_data")), original
            )
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

    def test_upload_database_and_lease_failures_do_not_delete_previous_data(self):
        self.initialize()
        checkpoint = Checkpoint.objects.get()
        rows, states = replay(
            self.seed, self.terminal, self.groups, self.start, self.end
        )
        for target in ("storage", "database", "lease"):
            patches = {
                "storage": patch.object(
                    default_storage, "save", side_effect=RuntimeError("upload failed")
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

    def test_checkpoint_deletion_only_removes_file_after_commit(self):
        self.initialize()
        checkpoint = Checkpoint.objects.get()
        name = checkpoint.data.name
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
            names[boundary] = Checkpoint.objects.get(timestamp=boundary).data.name
        expected = [self.start + pd.Timedelta(minutes=i) for i in (0, 1, 3, 4)]
        self.assertEqual(
            list(
                Checkpoint.objects.order_by("timestamp").values_list(
                    "timestamp", flat=True
                )
            ),
            expected,
        )
        self.assertTrue(default_storage.exists(original.data.name))
        for boundary, name in names.items():
            self.assertEqual(default_storage.exists(name), boundary in expected)

    def test_bad_checksum_and_earlier_history_are_not_silently_reseeded(self):
        self.initialize()
        checkpoint = Checkpoint.objects.get()
        checkpoint.state["sha256"] = "bad"
        checkpoint.save(update_fields=["state"])
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
        OrderBookData.objects.bulk_create(
            [
                OrderBookData(symbol=self.symbol, timestamp=moment, json_data=values)
                for moment, values in rows
            ]
        )
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


class QuickNodeBookAdapterTest(BookFixture, SimpleTestCase):
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
