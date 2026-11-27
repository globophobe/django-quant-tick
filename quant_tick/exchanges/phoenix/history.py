"""Reconstruct hour-end positions backwards from an atomic account snapshot.

The active trader buffer selects executions and position transfers, excluding
price-only updates. Every position's after-state, market trade sequence and OI
checkpoint must reconcile before undoing it. Unknown or missing history raises;
neither native OI nor aggregate account deltas can substitute for account state.
Before replay, the indexed head must match the latest successful transaction in
the native finalized ledger through the snapshot slot. A snapshot awaiting index
catch-up is stored separately and cannot be used as reconstructed hour-end state.

Checkpoints retain full replay state in compressed files, with cursor metadata
and a checksum in the database. Existing inline states move to files when reused.
Requested hours commit checkpoints with their statistics; intervening hours can
save progress independently without publishing an observation. Hourly cohorts use
the native bucket's mark price for size tiers, not the present snapshot's price.
Exported position flows include transfers and forced closes and reconcile both
hour boundaries; taker flow and liquidation executions remain trade-only.
Minute-sampled OI includes every replayed position change, including transfers,
ADL and shutdowns. It uses exact hour boundaries, independently of native sampled OI.
"""

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from itertools import chain

import pandas as pd
from django.core.files.base import ContentFile
from django.db import transaction

from quant_tick.constants import CheckpointType
from quant_tick.models import Checkpoint
from quant_tick.signals.files import discard

from ..position_metrics import minute_open_interest
from ..quicknode.solana import address_transactions, latest_address_signature
from .accounts import cohort_values
from .events import NONTRADE_EVENTS, PROGRAM, transaction_events
from .positioning import nontrade_flows, phoenix_positioning
from .snapshots import fetch_position_snapshot


def validate_transfers(group):
    expected, observed = Counter(), Counter()
    assets = set()
    for item in group:
        kind = item["kind"]
        if kind == "LiquidationTransfer":
            asset, quantity = item["asset_id"], item["base_lots_transferred"]
            expected[asset, item["liquidatee"]] -= quantity
            expected[asset, item["liquidator"]] += quantity
            assets.add(asset)
        elif kind in ("CloseMatchedPositions", "ShutdownClosePositions"):
            asset, quantity = item["asset_id"], abs(item["base_lots_closed"])
            long = (
                item["closed_long"]
                if kind == "CloseMatchedPositions"
                else item["long_trader"]
            )
            short = (
                item["closed_short"]
                if kind == "CloseMatchedPositions"
                else item["short_trader"]
            )
            expected[asset, long] -= quantity
            expected[asset, short] += quantity
            assets.add(asset)
    if not assets:
        return
    for item in group:
        if item["kind"] == "PnL" and item["asset_id"] in assets:
            observed[item["asset_id"], item["trader"]] += (
                item["base_lots_after"] - item["base_lots_before"]
            )
    if {k: v for k, v in expected.items() if v} != {
        k: v for k, v in observed.items() if v
    }:
        raise ValueError("Phoenix transfer/close disagrees with its position changes")


def decode_transaction(tx, address):
    message, meta = tx["transaction"]["message"], tx["meta"]
    if (
        meta is None
        or meta.get("err") is not None
        or meta.get("innerInstructions") is None
    ):
        raise ValueError("Missing or failed Phoenix history transaction metadata")
    loaded = meta.get("loadedAddresses", {})
    keys = (
        message["accountKeys"] + loaded.get("writable", []) + loaded.get("readonly", [])
    )
    if address not in keys:
        raise ValueError("Phoenix transaction does not contain the history account")
    instructions = message["instructions"] + [
        ix for group in meta["innerInstructions"] for ix in group["instructions"]
    ]
    if not any(keys[ix["programIdIndex"]] == PROGRAM for ix in instructions):
        return [], None
    groups = transaction_events(tx, tx["transaction"]["signatures"][0])
    clocks = {
        item["timestamp"]
        for group in groups
        for item in group
        if item["kind"] == "SlotContext"
    }
    if len(clocks) != 1:
        raise ValueError("Phoenix history lacks a unique execution timestamp")
    return groups, clocks.pop()


class ReversePositions:
    def __init__(self, state):
        self.state = deepcopy(state)
        if self.state.get("version") != 1:
            raise ValueError("Unsupported Phoenix position checkpoint version")
        self.positions = {
            bytes.fromhex(key): value for key, value in self.state["positions"].items()
        }
        if any(
            len(key) != 32 or type(value) is not int or not value
            for key, value in self.positions.items()
        ):
            raise ValueError("Invalid Phoenix checkpoint positions")
        self.open_interest()

    def open_interest(self):
        long = sum(max(size, 0) for size in self.positions.values())
        short = sum(max(-size, 0) for size in self.positions.values())
        if long != short:
            raise ValueError("Phoenix reconstructed long and short OI disagree")
        return long

    def dump(self):
        return {
            **self.state,
            "positions": {key.hex(): size for key, size in self.positions.items()},
        }

    def market_events(self, groups):
        records = []
        for group in groups:
            validate_transfers(group)
            header = None
            for item in group:
                kind = item["kind"]
                if kind == "Header":
                    header = item
                    if (
                        header["asset_id"] == self.state["asset_id"]
                        and header["base_lot_decimals"]
                        != self.state["base_lot_decimals"]
                    ):
                        raise ValueError("Phoenix historical market units changed")
                asset = item.get("asset_id")
                if kind == "TradeSummary":
                    if header is None:
                        raise ValueError("Phoenix trade lacks its market header")
                    asset = header["asset_id"]
                if asset == self.state["asset_id"]:
                    records.append(item)
        return records

    def undo(self, tx, groups, clock):
        if (
            clock is not None
            and self.state["clock"] is not None
            and clock > self.state["clock"]
        ):
            raise ValueError(
                "Phoenix history clock moved forward during reverse replay"
            )
        for item in reversed(self.market_events(groups)):
            kind = item["kind"]
            if kind == "PnL":
                trader = item["trader"]
                if self.positions.get(trader, 0) != item["base_lots_after"]:
                    raise ValueError("Phoenix reverse position continuity failed")
                if item["base_lots_before"]:
                    self.positions[trader] = item["base_lots_before"]
                else:
                    self.positions.pop(trader, None)
            elif kind == "MarketSummary":
                if self.open_interest() != item["open_interest"]:
                    raise ValueError("Phoenix reverse OI checkpoint mismatch")
            elif kind in ("TradeSummary", "ShutdownClosePositions"):
                sequence = item["trade_sequence_number"]
                expected = self.state.get("trade_sequence")
                previous_slot = self.state.get("trade_slot")
                if (expected is not None and sequence != expected) or (
                    previous_slot is not None and previous_slot != tx["slot"]
                ):
                    raise ValueError("Phoenix reverse trade sequence has a gap")
                self.state["trade_sequence"] = sequence - 1
                self.state["trade_slot"] = item.get("prev_trade_sequence_number_slot")
        self.open_interest()
        self.state.update(
            slot=tx["slot"],
            transaction_index=tx["transactionIndex"],
            before=tx["transaction"]["signatures"][0],
        )
        if clock is not None:
            self.state["clock"] = clock


class PhoenixPositionHistory:
    def __init__(self, symbol, *, assert_lease_owned=None):
        self.symbol = symbol
        self.assert_lease_owned = assert_lease_owned
        self.replay = None
        self.pending = []
        self.next_transaction = None
        self.snapshot_id = None

    def save(self, timestamp, state, *, checkpoint_type=None):
        with ExitStack() as cleanup:
            with transaction.atomic():
                checkpoint = self._save(timestamp, state, checkpoint_type, cleanup)
            cleanup.pop_all()
        if checkpoint_type is None:
            self.snapshot_id = None
        return checkpoint

    def _save(self, timestamp, state, checkpoint_type, cleanup):
        if self.assert_lease_owned is not None:
            self.assert_lease_owned()
        checkpoint, created = Checkpoint.objects.select_for_update().get_or_create(
            symbol=self.symbol,
            checkpoint_type=checkpoint_type or CheckpointType.PHOENIX_POSITIONS,
            timestamp=timestamp,
            defaults={"json_data": {}},
        )
        stored = state if created else checkpoint.read_state()
        if not created and any(
            stored[key] != state[key]
            for key in (
                "version",
                "positions",
                "asset_id",
                "base_lot_decimals",
                "history_address",
            )
        ):
            raise ValueError(
                "Phoenix replay disagrees with a saved position checkpoint"
            )
        if not checkpoint.file_data:
            raw = json.dumps(stored, separators=(",", ":")).encode()
            checkpoint.file_data.save(
                "state.json.gz", ContentFile(gzip.compress(raw, mtime=0)), save=False
            )
            cleanup.callback(
                discard, checkpoint.file_data.storage, checkpoint.file_data.name
            )
            checkpoint.json_data = {
                **{key: value for key, value in stored.items() if key != "positions"},
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            if self.assert_lease_owned is not None:
                self.assert_lease_owned()
            checkpoint.save(update_fields=["file_data", "json_data"])
        if checkpoint_type is None and self.snapshot_id is not None:
            Checkpoint.objects.filter(
                pk=self.snapshot_id, checkpoint_type=CheckpointType.PHOENIX_SNAPSHOT
            ).delete()
        return checkpoint

    def _checkpoint(self, end):
        checkpoint = (
            Checkpoint.objects.filter(
                symbol=self.symbol,
                checkpoint_type=CheckpointType.PHOENIX_POSITIONS,
                timestamp__gte=end,
            )
            .order_by("timestamp")
            .first()
        )
        if checkpoint is None:
            checkpoint = (
                Checkpoint.objects.filter(
                    symbol=self.symbol,
                    checkpoint_type=CheckpointType.PHOENIX_SNAPSHOT,
                    timestamp__gt=end,
                )
                .order_by("timestamp")
                .first()
            )
        if checkpoint is None:
            snapshot, block_time, _ = fetch_position_snapshot(self.symbol.api_symbol)
            state = {
                "version": 1,
                "slot": snapshot["slot"],
                "clock": None,
                "asset_id": snapshot["asset_id"],
                "base_lot_decimals": snapshot["base_lot_decimals"],
                "history_address": snapshot["history_address"],
                "positions": {
                    key.hex(): value for key, value in snapshot["positions"].items()
                },
            }
            if block_time <= end:
                raise ValueError(
                    "Phoenix snapshot does not cover the requested hour end"
                )
            state["history_head"] = latest_address_signature(
                state["history_address"], state["slot"]
            )
            checkpoint = self.save(
                block_time, state, checkpoint_type=CheckpointType.PHOENIX_SNAPSHOT
            )
        return checkpoint

    def load(self, end):
        while True:
            selected = self._checkpoint(end)
            with ExitStack() as cleanup:
                with transaction.atomic():
                    if self.assert_lease_owned is not None:
                        self.assert_lease_owned()
                    # Deletion removes the file on commit; hold its row through
                    # the read and any conversion of an existing inline state.
                    checkpoint = (
                        Checkpoint.objects.select_for_update()
                        .filter(pk=selected.pk)
                        .first()
                    )
                    if checkpoint is None:
                        continue
                    state = checkpoint.read_state()
                    if not state.get("history_head"):
                        raise ValueError("Phoenix checkpoint lacks its ledger boundary")
                    replay = ReversePositions(state)
                    if not checkpoint.file_data:
                        self._save(
                            checkpoint.timestamp,
                            state,
                            checkpoint.checkpoint_type,
                            cleanup,
                        )
                cleanup.pop_all()
            break
        self.boundary = checkpoint.timestamp
        self.transactions = iter(
            address_transactions(
                state["history_address"],
                state["slot"],
                before=state.get("before"),
                transaction_index=state.get("transaction_index"),
            )
        )
        if checkpoint.checkpoint_type == CheckpointType.PHOENIX_SNAPSHOT:
            self.snapshot_id = checkpoint.pk
            first = next(self.transactions, None)
            head = state["history_head"]
            if (
                first is None
                or first["slot"] != head["slot"]
                or first["transaction"]["signatures"][0] != head["signature"]
            ):
                raise ValueError(
                    "Phoenix indexed history has not reached the snapshot ledger boundary"
                )
            self.transactions = chain((first,), self.transactions)
        self.replay = replay

    def peek(self):
        while self.next_transaction is None:
            try:
                tx = next(self.transactions)
            except StopIteration:
                raise ValueError(
                    "Phoenix transaction history ended before the requested boundary"
                ) from None
            groups, clock = decode_transaction(tx, self.replay.state["history_address"])
            if clock is None:
                self.replay.undo(tx, groups, clock)
            else:
                if (
                    self.replay.state["clock"] is not None
                    and clock > self.replay.state["clock"]
                ):
                    raise ValueError("Phoenix history clock is not newest first")
                self.next_transaction = tx, groups, clock
        return self.next_transaction

    def rewind(self, boundary, *, collect=False):
        records = {}
        forced = []
        changes = defaultdict(Decimal)
        base = Decimal(10) ** -self.replay.state["base_lot_decimals"]
        limit = int(boundary.timestamp())
        while True:
            tx, groups, clock = self.peek()
            if clock < limit:
                self.boundary = boundary
                return records, forced, changes
            if collect:
                events = self.replay.market_events(groups)
                if any(item["kind"] == "TradeSummary" for item in events):
                    records[tx["transaction"]["signatures"][0]] = groups
                changes[clock // 60] += (
                    sum(
                        max(item["base_lots_after"], 0)
                        - max(item["base_lots_before"], 0)
                        for item in events
                        if item["kind"] == "PnL"
                    )
                    * base
                )
                forced.extend(
                    group
                    for group in groups
                    if any(
                        item["kind"] in NONTRADE_EVENTS
                        and item["asset_id"] == self.replay.state["asset_id"]
                        for item in group
                    )
                )
            self.replay.undo(tx, groups, clock)
            self.next_transaction = None

    def enrich(self, row):
        start = row["timestamp"].to_pydatetime()
        end = start + timedelta(hours=1)
        if self.replay is None:
            self.load(end)
        if end > self.boundary:
            raise ValueError("Phoenix reconstruction requires newest-first hours")
        while self.boundary > end:
            boundary = max(
                end,
                (
                    pd.Timestamp(self.boundary).ceil("h") - pd.Timedelta(hours=1)
                ).to_pydatetime(),
            )
            self.rewind(boundary)
            self.save(boundary, self.replay.dump())
        end_state = self.replay.dump()
        values = cohort_values(
            {
                **self.replay.state,
                "positions": self.replay.positions,
                "open_interest_lots": self.replay.open_interest(),
                "price": row["mark_price"],
            }
        )
        values.update(
            cohort_source="solana_position_replay",
            cohort_price_source="native_hourly_mark",
            cohort_timestamp=end,
            cohort_available_at=end,
            cohort_replay_validated=True,
            cohort_transaction_index=self.replay.state.get("transaction_index"),
        )
        end_oi = self.replay.open_interest()
        groups, forced, changes = self.rewind(start, collect=True)
        flows = phoenix_positioning(
            self.symbol.api_symbol.upper(), start, end, replayed_groups=groups
        )
        for key, value in nontrade_flows(
            forced,
            self.replay.state["asset_id"],
            self.replay.state["base_lot_decimals"],
        ).items():
            flows[key] = flows.get(key, 0) + value
        base = Decimal(10) ** -self.replay.state["base_lot_decimals"]
        for direction, sign in (("long", 1), ("short", -1)):
            before = [
                sign * size
                for size in self.replay.positions.values()
                if sign * size > 0
            ]
            after = [
                sign * size
                for size in end_state["positions"].values()
                if sign * size > 0
            ]
            if sum(before) * base + flows[direction + "_open_size"] - flows[
                direction + "_close_size"
            ] != sum(after) * base or len(before) + flows[
                direction + "_account_delta"
            ] != len(after):
                raise ValueError(
                    "Phoenix position flows do not reconcile with hour boundaries"
                )
        flows.update(
            minute_open_interest(
                self.replay.open_interest() * base, end_oi * base, changes
            )
        )
        flows["liquidation_flow_validated"] = True
        self.pending = [(end, end_state), (start, self.replay.dump())]
        return {**flows, **values}

    def commit(self, write_statistics):
        """Commit statistics and checkpoints together; discard uploads on rollback."""
        with ExitStack() as cleanup:
            with transaction.atomic():
                write_statistics()
                for timestamp, state in self.pending:
                    self._save(timestamp, state, None, cleanup)
            cleanup.pop_all()
        self.snapshot_id = None
        self.pending = []
