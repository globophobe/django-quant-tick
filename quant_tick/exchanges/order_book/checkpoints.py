"""Immutable compressed book checkpoints; database rows retain small metadata."""

import gzip
import hashlib
import json
import logging

from django.core.files.base import ContentFile
from django.db import transaction

from quant_tick.exchanges.hyperliquid.api import normalize_coin
from quant_tick.models import Checkpoint, OrderBookData, Symbol

from .book import Book

logger = logging.getLogger(__name__)
KIND = "hyperliquid_order_book"


def encode(state):
    raw = json.dumps(state, separators=(",", ":"), sort_keys=True).encode()
    meta = {
        key: state[key]
        for key in ("version", "coin", "block_number", "timestamp", "boundary")
    }
    meta.update(orders=len(state["orders"]), sha256=hashlib.sha256(raw).hexdigest())
    return meta, gzip.compress(raw, mtime=0)


def load(checkpoint):
    if not checkpoint.data.name:
        raise ValueError("Order book checkpoint has no stored snapshot")
    with checkpoint.data.open("rb") as source, gzip.GzipFile(fileobj=source) as zipped:
        raw = zipped.read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024 or hashlib.sha256(
        raw
    ).hexdigest() != checkpoint.state.get("sha256"):
        raise ValueError("Order book checkpoint checksum mismatch")
    state = json.loads(raw)
    meta, _ = encode(state)
    if meta != checkpoint.state or len(state["orders"]) != checkpoint.state["orders"]:
        raise ValueError("Order book checkpoint metadata mismatch")
    book = Book(state, normalize_coin(checkpoint.symbol.api_symbol))
    if book.time != checkpoint.timestamp:
        raise ValueError("Order book checkpoint timestamp mismatch")
    return state


def discard(file):
    try:
        file.storage.delete(file.name)
    except Exception:
        logger.exception("Order book snapshot cleanup failed: %s", file.name)


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
                symbol=symbol, kind=KIND, timestamp=boundary, state=meta
            )
            checkpoint.data.save("snapshot.json.gz", ContentFile(data), save=False)
            staged.append(checkpoint)
        with transaction.atomic():
            if assert_lease_owned:
                assert_lease_owned()
            Symbol.objects.select_for_update().get(pk=symbol.pk)
            for checkpoint in staged:
                existing = Checkpoint.objects.filter(
                    symbol=symbol, kind=KIND, timestamp=checkpoint.timestamp
                ).first()
                if existing:
                    if existing.state != checkpoint.state:
                        raise ValueError(
                            "Replayed order book disagrees with a saved checkpoint"
                        )
                    transaction.on_commit(lambda file=checkpoint.data: discard(file))
                else:
                    checkpoint.save(force_insert=True)
            OrderBookData.objects.bulk_create(
                [
                    OrderBookData(symbol=symbol, timestamp=moment, json_data=values)
                    for moment, values in rows
                ],
                update_conflicts=True,
                update_fields=["json_data"],
                unique_fields=["symbol", "timestamp"],
                batch_size=1000,
            )
            checkpoints = Checkpoint.objects.filter(symbol=symbol, kind=KIND).order_by(
                "timestamp"
            )
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
            discard(checkpoint.data)
        raise
