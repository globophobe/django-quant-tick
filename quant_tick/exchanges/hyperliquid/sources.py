"""Optional historical providers keep the native Hyperliquid market identity."""

import os
import re
from functools import lru_cache

from quant_tick.lib import get_current_time, parse_datetime

from ..coinapi.api import get_coinapi_response
from .api import normalize_coin


def coinapi_symbol(api_symbol):
    coin = normalize_coin(api_symbol)
    if not os.environ.get("COINAPI_API_KEY", "").strip():
        return None
    # HIP-3 and spot IDs use different vendor mappings and quantity contracts.
    if not re.fullmatch(r"[A-Za-z0-9]+", coin):
        return None
    return "HYPERLIQUID_PERP_" + coin.upper() + "_USDC"


def coinapi_history_start(api_symbol):
    return _coinapi_history_start(api_symbol, get_current_time().date())


@lru_cache(maxsize=128)
def _coinapi_history_start(api_symbol, day):
    """Refresh public coverage daily; never scan paid history before it begins."""
    symbols = get_coinapi_response(
        "/v1/symbols/HYPERLIQUID/active", {"filter_symbol_id": api_symbol}
    )
    matches = [row for row in symbols if row["symbol_id"] == api_symbol]
    if len(matches) > 1:
        raise ValueError("CoinAPI returned duplicate symbol metadata")
    if not matches or not matches[0].get("data_trade_start"):
        return None
    return parse_datetime(matches[0]["data_trade_start"])


def femtos_available():
    return bool(os.environ.get("FEMTOS_API_KEY", "").strip())
