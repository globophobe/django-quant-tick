"""Public Solana RPC using Build's 50 req/s and getProgramAccounts' 3 req/s limits."""

import os
from functools import lru_cache
from urllib.parse import urlsplit

from django.core.exceptions import ImproperlyConfigured

from ..public import get_public_json
from .rpc_errors import check_rpc_response


def available():
    return bool(os.environ.get("QUICKNODE_SOLANA_URL", "").strip())


def rpc(method, params, *, request_id=1):
    url = os.environ.get("QUICKNODE_SOLANA_URL", "").strip()
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not (parts.hostname or "").endswith(".quiknode.pro")
        or not parts.path.strip("/")
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise ImproperlyConfigured(
            "Set QUICKNODE_SOLANA_URL to the full HTTPS endpoint"
        )
    rate = 3 if method == "getProgramAccounts" else 50
    payload = get_public_json(
        url,
        "",
        {},
        namespace="QUICKNODE_SOLANA",
        max_requests=1,
        reset_seconds=1 / rate,
        redact_url=True,
        check_response=check_rpc_response,
        json_body={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        },
    )
    if (
        not isinstance(payload, dict)
        or payload.get("id") != request_id
        or "error" in payload
        or payload.get("result") is None
    ):
        raise ValueError("QuickNode omitted or rejected the requested Solana result")
    return payload["result"]


@lru_cache(maxsize=2048)
def transaction(signature):
    result = rpc(
        "getTransaction",
        [
            signature,
            {
                "encoding": "json",
                "commitment": "finalized",
                "maxSupportedTransactionVersion": 1,
            },
        ],
        request_id=signature,
    )
    if not isinstance(result, dict) or not result:
        raise ValueError("QuickNode returned an invalid Solana transaction")
    return result


def latest_address_signature(address, slot):
    """Find the last successful ledger transaction at or before a snapshot slot.

    minContextSlot fences the native RPC node, not the historical result range.
    Newer slots and failed transactions must be passed before selecting the head.
    """
    options = {"commitment": "finalized", "minContextSlot": slot, "limit": 1000}
    previous_slot, seen = None, set()
    while True:
        page = rpc("getSignaturesForAddress", [address, options])
        if not isinstance(page, list):
            raise TypeError("Invalid Solana signature history page")
        for item in page:
            signature, item_slot = item.get("signature"), item.get("slot")
            if (
                not isinstance(signature, str)
                or not signature
                or signature in seen
                or type(item_slot) is not int
                or item_slot < 0
                or (previous_slot is not None and item_slot > previous_slot)
                or item.get("confirmationStatus") != "finalized"
                or "err" not in item
            ):
                raise ValueError("Invalid or unordered finalized Solana signatures")
            previous_slot = item_slot
            seen.add(signature)
            if item_slot <= slot and item["err"] is None:
                return {"signature": signature, "slot": item_slot}
        if len(page) < options["limit"]:
            raise ValueError("Solana ledger has no successful history for the snapshot")
        options["before"] = page[-1]["signature"]


def address_transactions(address, slot, *, before=None, transaction_index=None):
    """Stream finalized transactions backwards, preserving the venue's order.

    A saved signature is an exclusive boundary within its slot. Provider cursors
    are used only between pages of that fixed request, never across snapshots.
    """
    options = {
        "transactionDetails": "full",
        "sortOrder": "desc",
        "limit": 1000,
        "encoding": "json",
        "maxSupportedTransactionVersion": 1,
        "commitment": "finalized",
        "filters": {"slot": {"lte": slot}, "status": "succeeded"},
    }
    if before is not None:
        options["filters"]["signature"] = {"lt": before}
    previous = (slot, transaction_index) if transaction_index is not None else None
    while True:
        result = rpc("getTransactionsForAddress", [address, options])
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise TypeError("Invalid Solana transaction history page")
        for tx in result["data"]:
            order = tx.get("slot"), tx.get("transactionIndex")
            if (
                any(type(value) is not int or value < 0 for value in order)
                or order[0] > slot
                or (previous is not None and order >= previous)
            ):
                raise ValueError(
                    "Solana transaction history is not strictly newest first"
                )
            previous = order
            yield tx
        token = result.get("paginationToken")
        if token is None:
            return
        if (
            not result["data"]
            or token != f"{previous[0]}:{previous[1]}"
            or token == options.get("paginationToken")
        ):
            raise ValueError("Solana transaction history cursor did not advance")
        options["paginationToken"] = token
