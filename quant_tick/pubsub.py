from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.api import candles_api
from quant_tick.models import Symbol, TradeData

logger = logging.getLogger(__name__)

RAW_TRADES = "raw-trades"
AGGREGATED_TRADES = "aggregated-trades"
SIGNIFICANT_TRADES = "significant-trades"
SIGNIFICANT_TRADE_FILTER_ATTR = "significant_trade_filter"
TRADE_STREAMS = (RAW_TRADES, AGGREGATED_TRADES, SIGNIFICANT_TRADES)
PUBSUB_PULL_BATCH_SIZE = 1000
PUBSUB_PULL_TIMEOUT = 1
DECIMAL_FIELDS = (
    "price",
    "volume",
    "notional",
    "high",
    "low",
    "totalBuyVolume",
    "totalVolume",
    "totalBuyNotional",
    "totalNotional",
)
INT_FIELDS = (
    "nanoseconds",
    "tickRule",
    "ticks",
    "totalBuyTicks",
    "totalTicks",
)


@dataclass(frozen=True)
class PubSubIngestionResult:
    pulled: int = 0
    processed: int = 0
    ok: int = 0
    failed: int = 0
    pending: int = 0
    ignored: int = 0

    @property
    def needs_rest_retry(self) -> bool:
        return self.failed > 0


def get_subscriber_client():
    """Create a pub/sub subscriber if enabled."""
    try:
        from google.cloud import pubsub_v1
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-pubsub is required for pub/sub. "
            "Install with django-quant-tick[pubsub] or google-cloud-pubsub."
        ) from exc
    return pubsub_v1.SubscriberClient()


def pubsub_name_part(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def get_trade_pubsub_configs(
    symbol: Symbol,
    stream: str | None = None,
) -> list[tuple[str, str]]:
    """Get configured trade Pub/Sub subscriptions for a symbol."""
    if stream is not None:
        if stream not in TRADE_STREAMS:
            raise ValueError(f"Unsupported trade pub/sub: {stream}.")
        subscription = get_trade_subscription_path(stream, symbol)
        return [(stream, subscription)] if subscription else []
    return [
        (stream, subscription)
        for stream in TRADE_STREAMS
        if (subscription := get_trade_subscription_path(stream, symbol))
    ]


def get_default_project_id() -> str:
    try:
        import google.auth
    except ImportError:
        return ""
    try:
        _credentials, project_id = google.auth.default()
    except Exception:
        return ""
    return project_id or ""


def symbol_supports_trade_stream(symbol: Symbol | None, stream: str) -> bool:
    if symbol is None:
        return False
    if stream == RAW_TRADES:
        return symbol.save_raw
    if stream == AGGREGATED_TRADES:
        return symbol.save_aggregated
    if stream == SIGNIFICANT_TRADES:
        return bool(symbol.significant_trade_filter)
    raise ValueError(f"Unsupported trade Pub/Sub stream: {stream}.")


def get_trade_subscription_id(stream: str, symbol: Symbol) -> str:
    if stream not in TRADE_STREAMS:
        raise ValueError(f"Unsupported trade Pub/Sub stream: {stream}.")
    subscription_id = (
        f"{pubsub_name_part(symbol.exchange)}-"
        f"{pubsub_name_part(symbol.api_symbol)}-{stream}"
    )
    if stream == SIGNIFICANT_TRADES:
        subscription_id += f"-{symbol.significant_trade_filter}"
    return subscription_id


def get_trade_subscription_path(stream: str, symbol: Symbol | None) -> str:
    if not symbol_supports_trade_stream(symbol, stream):
        return ""
    project_id = get_default_project_id()
    if not project_id:
        return ""
    subscription_id = get_trade_subscription_id(stream, symbol)
    return f"projects/{project_id}/subscriptions/{subscription_id}"


def get_trade_subscription_filter(stream: str, symbol: Symbol) -> str:
    if stream not in TRADE_STREAMS:
        raise ValueError(f"Unsupported trade Pub/Sub stream: {stream}.")
    filter_expr = (
        f'attributes.exchange="{symbol.exchange}" '
        f'AND attributes.symbol="{symbol.api_symbol}"'
    )
    if stream == SIGNIFICANT_TRADES:
        filter_expr += (
            f' AND attributes.{SIGNIFICANT_TRADE_FILTER_ATTR}='
            f'"{symbol.significant_trade_filter}"'
        )
    return filter_expr


def parse_trade_data(data: bytes | str) -> dict:
    """Parse pub/sub trade data."""
    payload = json.loads(data)
    row = dict(payload)
    row["timestamp"] = parse_timestamp(row["timestamp"])
    for field in DECIMAL_FIELDS:
        if field in row:
            row[field] = to_decimal_or_none(row[field])
    for field in INT_FIELDS:
        if field in row:
            row[field] = to_int_or_none(row[field])
    return row


def parse_timestamp(value: str) -> datetime:
    timestamp = pd.Timestamp(value).to_pydatetime()
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def to_decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def trades_to_data_frame(rows: list[dict]) -> DataFrame:
    if not rows:
        return DataFrame([])
    frame = DataFrame(rows)
    return frame.sort_values(["timestamp", "nanoseconds", "uid"]).reset_index(drop=True)


def matches_significant_trade_filter(message, symbol: Symbol) -> bool:
    if not symbol.significant_trade_filter:
        return False
    attrs = dict(message.attributes)
    return attrs.get(SIGNIFICANT_TRADE_FILTER_ATTR) == str(
        symbol.significant_trade_filter
    )


def ingest_trades_from_pubsub(
    *,
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    configs: list[tuple[str, str]] | None = None,
    batch_size: int = PUBSUB_PULL_BATCH_SIZE,
    timeout: float = PUBSUB_PULL_TIMEOUT,
    subscriber=None,
    candle_fetcher=candles_api,
) -> PubSubIngestionResult:
    """Pull symbol trade Pub/Sub messages and write validated trade data."""
    configs = configs if configs is not None else get_trade_pubsub_configs(symbol)
    if not configs:
        return PubSubIngestionResult()
    for stream, _subscription in configs:
        if stream not in TRADE_STREAMS:
            raise ValueError(f"Unsupported trade Pub/Sub stream: {stream}.")

    subscriber = subscriber or get_subscriber_client()
    rows_by_stream = {stream: [] for stream, _subscription in configs}
    ack_ids_by_subscription = defaultdict(list)
    release_ack_ids_by_subscription = defaultdict(list)
    pulled = 0
    ignored = 0

    for stream, subscription in configs:
        while True:
            response = subscriber.pull(
                subscription=subscription,
                max_messages=batch_size,
                timeout=timeout,
            )
            received = list(response.received_messages)
            if not received:
                break
            pulled += len(received)
            for received_message in received:
                message = received_message.message
                attrs = dict(message.attributes)
                if (
                    attrs.get("exchange") != symbol.exchange
                    or attrs.get("symbol") != symbol.api_symbol
                ):
                    release_ack_ids_by_subscription[subscription].append(
                        received_message.ack_id
                    )
                    continue

                if stream == SIGNIFICANT_TRADES and not matches_significant_trade_filter(
                    message,
                    symbol,
                ):
                    ignored += 1
                    ack_ids_by_subscription[subscription].append(received_message.ack_id)
                    continue

                row = parse_trade_data(message.data)
                if row["timestamp"] < timestamp_from:
                    ignored += 1
                    ack_ids_by_subscription[subscription].append(received_message.ack_id)
                    continue
                if row["timestamp"] >= timestamp_to:
                    release_ack_ids_by_subscription[subscription].append(
                        received_message.ack_id
                    )
                    continue
                rows_by_stream[stream].append(
                    (received_message.ack_id, row, subscription)
                )

    raw_trades = trades_to_data_frame(
        [row for _ack_id, row, _subscription in rows_by_stream.get(RAW_TRADES, [])]
    )
    aggregated_trades = trades_to_data_frame(
        [
            row
            for _ack_id, row, _subscription in rows_by_stream.get(
                AGGREGATED_TRADES,
                [],
            )
        ]
    )
    filtered_trades = trades_to_data_frame(
        [
            row
            for _ack_id, row, _subscription in rows_by_stream.get(
                SIGNIFICANT_TRADES,
                [],
            )
        ]
    )
    raw_trades = raw_trades if len(raw_trades) else None
    aggregated_trades = aggregated_trades if len(aggregated_trades) else None
    filtered_trades = filtered_trades if len(filtered_trades) else None
    processed = 0
    ok = 0
    failed = 0
    pending = 0
    row_items = [
        item for stream_rows in rows_by_stream.values() for item in stream_rows
    ]
    if row_items:
        processed = len(row_items)
        timestamps = [row["timestamp"] for _ack_id, row, _subscription in row_items]
        write_from = pd.Timestamp(min(timestamps)).floor("min").to_pydatetime()
        write_to = (
            pd.Timestamp(max(timestamps)).floor("min").to_pydatetime()
            + pd.Timedelta("1min")
        )
        candles = candle_fetcher(symbol, write_from, write_to, resolution="1m")
        trade_data = TradeData.write(
            symbol,
            write_from,
            write_to,
            candles,
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )
        if any(row.ok is None for row in trade_data):
            pending = processed
            for ack_id, _row, subscription in row_items:
                release_ack_ids_by_subscription[subscription].append(ack_id)
        else:
            for ack_id, _row, subscription in row_items:
                ack_ids_by_subscription[subscription].append(ack_id)
            if all(row.ok for row in trade_data):
                ok = processed
            else:
                failed = processed

    for subscription, ack_ids in ack_ids_by_subscription.items():
        subscriber.acknowledge(subscription=subscription, ack_ids=ack_ids)
    for subscription, ack_ids in release_ack_ids_by_subscription.items():
        subscriber.modify_ack_deadline(
            subscription=subscription,
            ack_ids=ack_ids,
            ack_deadline_seconds=0,
        )

    return PubSubIngestionResult(
        pulled=pulled,
        processed=processed,
        ok=ok,
        failed=failed,
        pending=pending,
        ignored=ignored,
    )
