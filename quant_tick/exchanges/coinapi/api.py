"""Authenticated public market data.

COINAPI_RATE_LIMIT sets REST requests per minute (10 when unset or blank). All REST
endpoints share one allowance per process. Separate workers still consume
the same organization quota at the provider.
"""

import os
import re
from functools import lru_cache

from django.core.exceptions import ImproperlyConfigured

from quant_tick.constants import SymbolType
from quant_tick.lib import get_current_time, parse_datetime

from ..public import get_public_json


def get_coinapi_symbol_type(api_symbol):
    """Accept markets whose trade and candle quantities are in base units."""
    if re.fullmatch(r"[A-Z0-9_-]+_SPOT_[A-Z0-9]+_[A-Z0-9]+", api_symbol):
        return SymbolType.SPOT
    if re.fullmatch(
        r"(?:HYPERLIQUID_PERP_[A-Z0-9]+_USDC|BYBIT_PERP_[A-Z0-9]+_USDT)", api_symbol
    ):
        return SymbolType.PERPETUAL
    raise ValueError(
        "CoinAPI requires a spot, Hyperliquid USDC perpetual, or Bybit USDT perpetual symbol ID"
    )


def get_coinapi_key():
    api_key = os.environ.get("COINAPI_API_KEY", "").strip()
    if not api_key:
        raise ImproperlyConfigured("Set COINAPI_API_KEY to collect CoinAPI market data")
    return api_key


def get_coinapi_response(path, params):
    try:
        rate_limit = int(os.environ.get("COINAPI_RATE_LIMIT") or 10)
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


def coinapi_history_start(api_symbol):
    return _coinapi_history_start(api_symbol, get_current_time().date())


@lru_cache(maxsize=128)
def _coinapi_history_start(api_symbol, day):
    """Refresh public coverage daily; never scan paid history before it begins."""
    get_coinapi_symbol_type(api_symbol)
    exchange = api_symbol.split("_SPOT_")[0].split("_PERP_")[0]
    symbols = get_coinapi_response(
        f"/v1/symbols/{exchange}/active", {"filter_symbol_id": api_symbol}
    )
    matches = [row for row in symbols if row["symbol_id"] == api_symbol]
    if len(matches) > 1:
        raise ValueError("CoinAPI returned duplicate symbol metadata")
    if not matches or not matches[0].get("data_trade_start"):
        return None
    return parse_datetime(matches[0]["data_trade_start"])
