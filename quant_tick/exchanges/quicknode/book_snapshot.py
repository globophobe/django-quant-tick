"""One complete public order snapshot, cancelled immediately after receipt."""

import logging
import os
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import grpc
from django.core.exceptions import ImproperlyConfigured

from quant_tick.exchanges.order_book.book import VERSION, Book, from_milliseconds
from quant_tick.exchanges.public import retry_after_seconds

from . import book_pb2
from .api import coin_literal

logger = logging.getLogger(__name__)
RETRYABLE = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
}


def endpoint():
    parts = urlsplit(os.environ.get("QUICKNODE_HYPERCORE_URL", "").strip())
    path = parts.path.rstrip("/").split("/")
    if path[-1] in ("evm", "info", "nanoreth", "hypercore"):
        path.pop()
    if (
        parts.scheme != "https"
        or not (parts.hostname or "").endswith(".quiknode.pro")
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
        or len(path) != 2
        or not path[1]
    ):
        raise ImproperlyConfigured(
            "Set QUICKNODE_HYPERCORE_URL to a valid Hyperliquid endpoint"
        )
    return parts.hostname + ":10000", path[1]


def snapshot(coin, *, retry=30):
    coin_literal(coin)
    address, token = endpoint()
    status = None
    limit_reported = False
    for attempt in range(retry + 1):
        with grpc.secure_channel(
            address,
            grpc.ssl_channel_credentials(),
            options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
        ) as channel:
            request = channel.unary_stream(
                "/hyperliquid.OrderBookStreaming/StreamL4BookUpdates",
                request_serializer=book_pb2.L4BookUpdatesRequest.SerializeToString,
                response_deserializer=book_pb2.L4BookUpdatesUpdate.FromString,
            )
            stream = request(
                book_pb2.L4BookUpdatesRequest(coins=[coin]),
                metadata=(("x-token", token),),
                timeout=30,
            )
            try:
                update = next(stream)
            except grpc.RpcError as error:
                status = error.code()
                headers = {
                    key.lower(): value
                    for key, value in (error.trailing_metadata() or ())
                }
                delay = retry_after_seconds(
                    SimpleNamespace(
                        headers={"Retry-After": headers.get("retry-after", "")}
                    ),
                    default=min(2**attempt, 30)
                    if status == grpc.StatusCode.RESOURCE_EXHAUSTED
                    else 1,
                )
            else:
                if not update.snapshot or not update.height or not update.time:
                    raise ValueError(
                        "QuickNode omitted the complete order book snapshot"
                    )
                orders = []
                for row in update.diffs:
                    if row.diff_type != 1 or row.coin != coin:
                        raise ValueError(
                            "QuickNode snapshot contains an unexpected market or update"
                        )
                    orders.append([row.oid, row.side, row.px, row.sz])
                state = {
                    "version": VERSION,
                    "coin": coin,
                    "block_number": update.height,
                    "timestamp": from_milliseconds(update.time).isoformat(),
                    "boundary": False,
                    "orders": orders,
                }
                book = Book(state, coin)
                logger.info(
                    "QuickNode %s: order book snapshot at block %s", coin, book.block
                )
                return book.state()
            finally:
                stream.cancel()
        if status == grpc.StatusCode.RESOURCE_EXHAUSTED and not limit_reported:
            logger.error(
                "QUICKNODE API credit or rate limit reached (gRPC RESOURCE_EXHAUSTED)"
            )
            limit_reported = True
        if status not in RETRYABLE or attempt == retry:
            break
        logger.info("QUICKNODE gRPC %s, sleeping %g seconds", status.name, delay)
        time.sleep(delay)
    # Raise outside the exception handler; gRPC errors can embed endpoint tokens.
    raise RuntimeError("QuickNode snapshot request failed: " + status.name)
