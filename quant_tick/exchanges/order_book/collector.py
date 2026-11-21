"""Forward order-book collection from saved full states.

The first run seeds the current book without emitting historical summaries.
Normal runs resume at the newest checkpoint. Retries start at the nearest
checkpoint at or before the requested range and replay through a later saved
checkpoint or a fresh live snapshot. History before the first seed is unavailable.
Summaries and new checkpoints are published only after terminal reconciliation.
"""

import logging

from quant_tick.constants import Exchange
from quant_tick.exchanges.hyperliquid.api import normalize_coin
from quant_tick.exchanges.quicknode.api import available as sql_available
from quant_tick.exchanges.quicknode.api import timestamp
from quant_tick.exchanges.quicknode.book_history import history
from quant_tick.exchanges.quicknode.book_snapshot import snapshot
from quant_tick.lib import get_current_time
from quant_tick.models import Checkpoint

from .checkpoints import KIND, encode, load, persist
from .replay import replay

logger = logging.getLogger(__name__)


def order_book(
    symbol, timestamp_from, timestamp_to, retry=False, *, assert_lease_owned=None
):
    if symbol.exchange != Exchange.HYPERLIQUID:
        raise NotImplementedError(
            "Order book collection is only implemented for Hyperliquid"
        )
    if not sql_available():
        raise ValueError("Set QUICKNODE_API_KEY to collect order book history")
    clamped = symbol.clamp_timestamp_range(timestamp_from, timestamp_to)
    if clamped is None:
        return
    requested_start, requested_end = map(timestamp, clamped)
    requested_end = min(requested_end, timestamp(get_current_time()).floor("min"))
    if requested_start > requested_end:
        return
    coin = normalize_coin(symbol.api_symbol)
    checkpoints = Checkpoint.objects.filter(symbol=symbol, kind=KIND).select_related(
        "symbol"
    )
    first = checkpoints.order_by("timestamp").first()
    if first is None:
        state = snapshot(coin)
        boundary = timestamp(state["timestamp"])
        persist(
            symbol, [], {boundary: encode(state)}, assert_lease_owned=assert_lease_owned
        )
        logger.info(
            "%s: order book initialized at %s; earlier history is unavailable",
            symbol,
            boundary,
        )
        return
    if requested_end <= first.timestamp:
        logger.warning("%s: order book history begins at %s", symbol, first.timestamp)
        return
    if retry:
        start = (
            checkpoints.filter(timestamp__lte=requested_start)
            .order_by("-timestamp")
            .first()
            or first
        )
        if requested_start < first.timestamp:
            logger.warning(
                "%s: order book history begins at %s", symbol, first.timestamp
            )
    else:
        start = (
            checkpoints.filter(timestamp__lte=requested_end)
            .order_by("-timestamp")
            .first()
        )
        if start.timestamp == requested_end:
            return
    state = load(start)
    terminal = (
        checkpoints.filter(timestamp__gte=requested_end).order_by("timestamp").first()
    )
    terminal_state = load(terminal) if terminal else snapshot(coin)
    if timestamp(terminal_state["timestamp"]) < requested_end:
        raise ValueError("Order book snapshot has not reached the requested end")
    if assert_lease_owned:
        assert_lease_owned()
    logger.info("%s: order book %s to %s", symbol, start.timestamp, requested_end)
    rows, states = replay(
        state,
        terminal_state,
        history(coin, state, terminal_state, assert_lease_owned=assert_lease_owned),
        max(requested_start, timestamp(start.timestamp)),
        requested_end,
    )
    persist(symbol, rows, states, assert_lease_owned=assert_lease_owned)
