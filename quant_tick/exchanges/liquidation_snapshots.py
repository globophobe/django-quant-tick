"""Persist full liquidation inputs before publishing their availability-hour summary."""

import gzip
import hashlib
import json
import logging
from functools import partial

import pandas as pd
from django.core.files.base import ContentFile
from django.db import transaction

from quant_tick.constants import CheckpointType, Exchange
from quant_tick.lib import get_current_time
from quant_tick.models import Checkpoint, PerpetualStatsData, Symbol
from quant_tick.signals.files import discard

from .hyperliquid.liquidation import fetch_snapshot as hyperliquid_snapshot
from .liquidation_exposure import VERSION
from .phoenix.liquidation import fetch_snapshot as phoenix_snapshot

logger = logging.getLogger(__name__)


def persist(symbol, state, *, assert_lease_owned=None):
    """Publish the immutable input file and summary together after upload succeeds."""
    available = pd.Timestamp(state["available_at"])
    hour = available.floor("h")
    raw = json.dumps(state, default=str, separators=(",", ":")).encode()
    summary = json.loads(json.dumps(state["summary"], default=str))
    meta = {
        "version": VERSION,
        "available_at": available.isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "summary": summary,
    }
    checkpoint = Checkpoint(
        symbol=symbol,
        checkpoint_type=CheckpointType.LIQUIDATION_EXPOSURE,
        timestamp=available,
        json_data=meta,
    )
    try:
        if assert_lease_owned:
            assert_lease_owned()
        checkpoint.file_data.save(
            "snapshot.json.gz", ContentFile(gzip.compress(raw, mtime=0)), save=False
        )
        with transaction.atomic():
            if assert_lease_owned:
                assert_lease_owned()
            Symbol.objects.select_for_update().get(pk=symbol.pk)
            existing = Checkpoint.objects.filter(
                symbol=symbol,
                checkpoint_type=CheckpointType.LIQUIDATION_EXPOSURE,
                timestamp__gte=hour,
                timestamp__lt=hour + pd.Timedelta(hours=1),
            ).first()
            if existing:
                summary = existing.json_data["summary"]
                transaction.on_commit(
                    partial(
                        discard, checkpoint.file_data.storage, checkpoint.file_data.name
                    )
                )
            else:
                checkpoint.save(force_insert=True)
            row, _ = PerpetualStatsData.objects.select_for_update().get_or_create(
                symbol=symbol, frequency=60, timestamp=hour
            )
            row.json_data = {**(row.json_data or {}), "liquidation_exposure": summary}
            row.save(update_fields=["json_data"])
            if assert_lease_owned:
                assert_lease_owned()
    except Exception:
        if checkpoint.file_data.name:
            discard(checkpoint.file_data.storage, checkpoint.file_data.name)
        raise


def collect(symbol, *, assert_lease_owned=None):
    adapters = {
        Exchange.HYPERLIQUID: hyperliquid_snapshot,
        Exchange.PHOENIX: phoenix_snapshot,
    }
    if symbol.exchange not in adapters or symbol.api_symbol != "BTC":
        return
    hour = pd.Timestamp(get_current_time()).floor("h")
    if Checkpoint.objects.filter(
        symbol=symbol,
        checkpoint_type=CheckpointType.LIQUIDATION_EXPOSURE,
        timestamp__gte=hour,
        timestamp__lt=hour + pd.Timedelta(hours=1),
    ).exists():
        return
    logger.info("%s: liquidation exposure %s", symbol, hour)
    state = adapters[symbol.exchange]()
    if state is not None:
        persist(symbol, state, assert_lease_owned=assert_lease_owned)
