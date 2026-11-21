"""Immutable compressed book checkpoints; database rows retain small metadata."""

import gzip
import hashlib
import json
from functools import partial

from django.core.files.base import ContentFile
from django.db import transaction

from quant_tick.constants import CheckpointType
from quant_tick.exchanges.hyperliquid.api import normalize_coin
from quant_tick.models import Checkpoint, Symbol
from quant_tick.signals.files import discard

from .book import Book
from .storage import write_minutes


def encode(state):
    raw = json.dumps(state, separators=(",", ":"), sort_keys=True).encode()
    meta = {
        key: state[key]
        for key in ("version", "coin", "block_number", "timestamp", "boundary")
    }
    meta.update(orders=len(state["orders"]), sha256=hashlib.sha256(raw).hexdigest())
    return meta, gzip.compress(raw, mtime=0)


def load(checkpoint):
    if not checkpoint.file_data.name:
        raise ValueError("Order book checkpoint has no stored snapshot")
    with (
        checkpoint.file_data.open("rb") as source,
        gzip.GzipFile(fileobj=source) as zipped,
    ):
        raw = zipped.read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024 or hashlib.sha256(
        raw
    ).hexdigest() != checkpoint.json_data.get("sha256"):
        raise ValueError("Order book checkpoint checksum mismatch")
    state = json.loads(raw)
    meta, _ = encode(state)
    if (
        meta != checkpoint.json_data
        or len(state["orders"]) != checkpoint.json_data["orders"]
    ):
        raise ValueError("Order book checkpoint metadata mismatch")
    book = Book(state, normalize_coin(checkpoint.symbol.api_symbol))
    if book.time != checkpoint.timestamp:
        raise ValueError("Order book checkpoint timestamp mismatch")
    return state


def persist(symbol, rows, states, *, assert_lease_owned=None):
    """Upload immutable files before the transaction that publishes their references.

    Failed transactions remove only their own uploads. Existing references and
    summary rows stay intact. Hourly anchors and the first seed are retained;
    only the two most recent intermediate checkpoints are kept for resuming.
    """
    staged = []
    try:
        for boundary, (meta, data) in states.items():
            if assert_lease_owned:
                assert_lease_owned()
            checkpoint = Checkpoint(
                symbol=symbol,
                checkpoint_type=CheckpointType.HYPERLIQUID_ORDER_BOOK,
                timestamp=boundary,
                json_data=meta,
            )
            checkpoint.file_data.save("snapshot.json.gz", ContentFile(data), save=False)
            staged.append(checkpoint)
        with transaction.atomic():
            if assert_lease_owned:
                assert_lease_owned()
            Symbol.objects.select_for_update().get(pk=symbol.pk)
            for checkpoint in staged:
                existing = Checkpoint.objects.filter(
                    symbol=symbol,
                    checkpoint_type=CheckpointType.HYPERLIQUID_ORDER_BOOK,
                    timestamp=checkpoint.timestamp,
                ).first()
                if existing:
                    if existing.json_data != checkpoint.json_data:
                        raise ValueError(
                            "Replayed order book disagrees with a saved checkpoint"
                        )
                    transaction.on_commit(
                        partial(
                            discard,
                            checkpoint.file_data.storage,
                            checkpoint.file_data.name,
                        )
                    )
                else:
                    checkpoint.save(force_insert=True)
            write_minutes(symbol, rows, staged)
            if assert_lease_owned:
                assert_lease_owned()
            checkpoints = Checkpoint.objects.filter(
                symbol=symbol, checkpoint_type=CheckpointType.HYPERLIQUID_ORDER_BOOK
            ).order_by("timestamp")
            first = checkpoints.values_list("pk", flat=True).first()
            intermediate = checkpoints.exclude(
                timestamp__minute=0, timestamp__second=0
            ).exclude(pk=first)
            keep = list(
                intermediate.order_by("-timestamp").values_list("pk", flat=True)[:2]
            )
            intermediate.exclude(pk__in=keep).delete()
    except Exception:
        for checkpoint in staged:
            discard(checkpoint.file_data.storage, checkpoint.file_data.name)
        raise
