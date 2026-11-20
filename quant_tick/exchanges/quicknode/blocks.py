"""Authoritative trade order for blocks whose SQL position chains are ambiguous."""

import os
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from django.core.exceptions import ImproperlyConfigured

from ..public import get_public_json
from .api import BORROW_LIQUIDATION_DIRECTIONS, number
from .rpc_errors import check_rpc_response


def get_trade_blocks(numbers):
    value = os.environ.get("QUICKNODE_HYPERCORE_URL", "").strip()
    if not value:
        raise ImproperlyConfigured(
            "Set QUICKNODE_HYPERCORE_URL to resolve ambiguous QuickNode trade blocks"
        )
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not (parts.hostname or "").endswith(".quiknode.pro")
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise ImproperlyConfigured("Invalid QUICKNODE_HYPERCORE_URL")
    path = parts.path.rstrip("/").split("/")
    if path[-1] in ("evm", "info", "nanoreth", "hypercore"):
        path.pop()
    url = urlunsplit(
        (parts.scheme, parts.netloc, "/".join(path) + "/hypercore", "", "")
    )
    found = {}
    for offset in range(0, len(numbers), 10):
        requested = numbers[offset : offset + 10]
        payload = [
            {
                "jsonrpc": "2.0",
                "id": block,
                "method": "hl_getBatchBlocks",
                "params": {"stream": "trades", "from": block, "to": block},
            }
            for block in requested
        ]
        results = get_public_json(
            url,
            "",
            {},
            namespace="QUICKNODE_BLOCKS",
            redact_url=True,
            check_response=check_rpc_response,
            json_body=payload,
        )
        if not isinstance(results, list) or len(results) != len(requested):
            raise ValueError("QuickNode returned an incomplete block batch")
        for result in results:
            if not isinstance(result, dict) or "error" in result:
                raise ValueError("QuickNode returned an invalid trade-block response")
            block = result.get("id")
            if block not in requested or block in found:
                raise ValueError("QuickNode returned an unexpected or duplicate block")
            blocks = result.get("result", {}).get("blocks", [])
            if len(blocks) != 1 or blocks[0].get("block_number") != block:
                raise ValueError("QuickNode omitted the requested trade block")
            found[block] = blocks[0]["events"]
    return found


def order_from_block(coin, rows, events):
    """Cross-check the complete SQL block, then follow native taker-fill order."""
    by_id = {row["trade_id"]: row for row in rows}
    ordered = []
    for _, fill in events:
        if fill.get("coin") != coin or not fill.get("crossed"):
            continue
        if fill.get("dir") in BORROW_LIQUIDATION_DIRECTIONS:
            continue
        trade_id = int(fill["tid"])
        # Dust allocations distribute an executed trade; they are not extra fills.
        if trade_id == 0 and fill.get("dir") == "Spot Dust Conversion":
            continue
        row = by_id.pop(trade_id, None)
        if row is None or (
            row["price"],
            row["size"],
            row["side"],
            row["timestamp"],
        ) != (
            number(fill["px"]),
            number(fill["sz"]),
            fill["side"],
            pd.Timestamp(int(fill["time"]), unit="ms", tz="UTC"),
        ):
            raise ValueError("QuickNode SQL trades and raw block disagree")
        ordered.append(row)
    if by_id:
        raise ValueError("QuickNode raw block omitted SQL trades")
    return ordered
