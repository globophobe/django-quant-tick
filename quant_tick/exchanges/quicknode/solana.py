"""Finalized public Solana transactions; endpoint credentials stay out of logs.

QUICKNODE_SOLANA_RATE_LIMIT defaults to ten requests per second. Solana RPC has
a separate allowance from SQL; both use the shared Retry-After handling.
"""

import os
from functools import lru_cache
from urllib.parse import urlsplit

from django.core.exceptions import ImproperlyConfigured

from ..public import get_public_json


def available():
    return bool(os.environ.get("QUICKNODE_SOLANA_URL", "").strip())


@lru_cache(maxsize=2048)
def transaction(signature):
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
    try:
        rate = int(os.environ.get("QUICKNODE_SOLANA_RATE_LIMIT") or 10)
        if rate <= 0:
            raise ValueError
    except ValueError as exc:
        raise ImproperlyConfigured(
            "QUICKNODE_SOLANA_RATE_LIMIT must be positive requests per second"
        ) from exc
    payload = get_public_json(
        url,
        "",
        {},
        namespace="QUICKNODE_SOLANA",
        max_requests=1,
        reset_seconds=1 / rate,
        retry=3,
        redact_url=True,
        json_body={
            "jsonrpc": "2.0",
            "id": signature,
            "method": "getTransaction",
            "params": [
                signature,
                {
                    "encoding": "json",
                    "commitment": "finalized",
                    "maxSupportedTransactionVersion": 1,
                },
            ],
        },
    )
    if (
        not isinstance(payload, dict)
        or payload.get("id") != signature
        or "error" in payload
        or not isinstance(payload.get("result"), dict)
    ):
        raise ValueError(
            "QuickNode omitted or rejected the requested Solana transaction"
        )
    return payload["result"]
