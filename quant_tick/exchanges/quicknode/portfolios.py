"""A bounded position census followed by current full-portfolio observations.

SQL identifies holders; it is not sufficient collateral evidence. The live RPC
supplies account modes and every DEX's state. Current states are never attached
to historical hours. Batch omissions remain explicit unknown accounts.
"""

import os
import re
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from django.core.exceptions import ImproperlyConfigured

from quant_tick.lib import get_current_time

from ..public import get_public_json
from .api import number, query, timestamp
from .rpc_errors import check_rpc_response

ACCOUNT_LIMIT = 1000


def position_census(limit=ACCOUNT_LIMIT):
    if not 0 < limit <= 10000:
        raise ValueError("Portfolio sample must fit the SQL result limit")
    rows = query(
        "SELECT block_number,snapshot_time FROM hyperliquid_oracle_prices "
        "WHERE coin='BTC' ORDER BY block_number DESC LIMIT 1"
    )
    if len(rows) != 1:
        raise ValueError("QuickNode omitted the position census boundary")
    anchor = rows[0]
    block = int(anchor["block_number"])
    moment = timestamp(anchor["snapshot_time"])
    anchor = {**anchor, "snapshot_time": moment.isoformat()}
    age = timestamp(get_current_time()) - moment
    if block <= 0 or not pd.Timedelta(0) <= age <= pd.Timedelta(minutes=30):
        raise ValueError("QuickNode position census is stale or in the future")
    totals = query(
        "SELECT sumIf(size,size>0) AS long_size,sumIf(-size,size<0) AS short_size,"
        "countIf(size>0) AS long_accounts,countIf(size<0) AS short_accounts "
        "FROM hyperliquid_clearinghouse_states FINAL "
        f"WHERE block_number={block} AND clearinghouse=0 AND coin='BTC' LIMIT 1"
    )
    if (
        len(totals) != 1
        or number(totals[0]["long_size"]) <= 0
        or number(totals[0]["long_size"]) != number(totals[0]["short_size"])
    ):
        raise ValueError("QuickNode position census does not balance")
    totals = totals[0]
    rows = query(
        "SELECT user,size FROM hyperliquid_clearinghouse_states FINAL "
        f"WHERE block_number={block} AND clearinghouse=0 AND coin='BTC' "
        f"AND size != 0 ORDER BY abs(size) DESC,user LIMIT {limit}"
    )
    if len(rows) != min(limit, totals["long_accounts"] + totals["short_accounts"]):
        raise ValueError("QuickNode returned an empty or truncated position census")
    positions = {}
    for row in rows:
        key = row["user"]
        if not re.fullmatch(r"0x[0-9a-f]{40}", key) or key in positions:
            raise ValueError("QuickNode returned an invalid or duplicate holder")
        positions[key] = number(row["size"])
        if not positions[key]:
            raise ValueError("QuickNode returned an empty position")
    return anchor, positions, totals


def portfolio_states(users):
    parts = urlsplit(os.environ.get("QUICKNODE_HYPERCORE_URL", "").strip())
    if (
        parts.scheme != "https"
        or not (parts.hostname or "").endswith(".quiknode.pro")
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise ImproperlyConfigured("Set QUICKNODE_HYPERCORE_URL to the HTTPS endpoint")
    path = parts.path.rstrip("/").split("/")
    if path[-1] in ("evm", "info", "nanoreth", "hypercore"):
        path.pop()
    if len(path) != 2 or not path[1]:
        raise ImproperlyConfigured("Invalid QUICKNODE_HYPERCORE_URL")
    url = urlunsplit(
        (parts.scheme, parts.netloc, "/".join(path) + "/hypercore", "", "")
    )
    states, failed = {}, []
    for offset in range(0, len(users), 500):
        requested = users[offset : offset + 500]
        payload = get_public_json(
            url,
            "",
            {},
            namespace="QUICKNODE_PORTFOLIOS",
            max_requests=50,
            redact_url=True,
            check_response=check_rpc_response,
            json_body={
                "jsonrpc": "2.0",
                "id": offset + 1,
                "method": "hl_batchPortfolioStates",
                "params": {"users": requested, "dex": "ALL_DEXES"},
            },
        )
        if (
            not isinstance(payload, dict)
            or payload.get("id") != offset + 1
            or "error" in payload
        ):
            raise ValueError("QuickNode rejected the portfolio batch")
        result = payload.get("result", {})
        success, missing = result.get("successful_states"), result.get("failed_wallets")
        if not isinstance(success, list) or not isinstance(missing, list):
            raise TypeError("QuickNode omitted portfolio batch outcomes")
        seen = set()
        for key, state in success:
            if key not in requested or key in seen or not isinstance(state, dict):
                raise ValueError("QuickNode returned an unexpected portfolio")
            seen.add(key)
            states[key] = state
        if len(set(missing)) != len(missing) or seen.intersection(missing):
            raise ValueError("QuickNode returned inconsistent portfolio outcomes")
        if seen.union(missing) != set(requested):
            raise ValueError("QuickNode omitted requested portfolio outcomes")
        failed.extend(missing)
    return states, failed
