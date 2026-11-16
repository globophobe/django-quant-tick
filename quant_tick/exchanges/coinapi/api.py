"""Authenticated public market data.

COINAPI_RATE_LIMIT sets REST requests per minute (default 10). All REST
endpoints share one allowance per process. Separate workers still consume
the same organization quota at the provider.
"""

import os
import re

from django.core.exceptions import ImproperlyConfigured

from quant_tick.constants import SymbolType

from ..public import get_public_json


def get_coinapi_symbol_type(api_symbol):
    """Accept markets whose trade and candle quantities are in base units."""
    if re.fullmatch(r"[A-Z0-9_-]+_SPOT_[A-Z0-9]+_[A-Z0-9]+", api_symbol):
        return SymbolType.SPOT
    if re.fullmatch(r"HYPERLIQUID_PERP_[A-Z0-9]+_USDC", api_symbol):
        return SymbolType.PERPETUAL
    raise ValueError(
        "CoinAPI requires a spot or Hyperliquid USDC perpetual symbol ID, "
        "e.g. COINBASE_SPOT_BTC_USD or HYPERLIQUID_PERP_BTC_USDC"
    )


def get_coinapi_key():
    api_key = os.environ.get("COINAPI_API_KEY", "").strip()
    if not api_key:
        raise ImproperlyConfigured("Set COINAPI_API_KEY to collect CoinAPI market data")
    return api_key


def get_coinapi_response(path, params):
    try:
        rate_limit = int(os.environ.get("COINAPI_RATE_LIMIT", "10"))
        if rate_limit <= 0:
            raise ValueError
    except ValueError as exc:
        raise ImproperlyConfigured(
            "COINAPI_RATE_LIMIT must be a positive integer (requests per minute)"
        ) from exc
    payload = get_public_json(
        "https://rest.coinapi.io",
        path,
        params,
        namespace="COINAPI",
        max_requests=1,
        reset_seconds=60 / rate_limit,
        headers={"X-CoinAPI-Key": get_coinapi_key()},
        retry=3,
    )
    if not isinstance(payload, list):
        raise TypeError("CoinAPI returned an unexpected market data response")
    return payload
