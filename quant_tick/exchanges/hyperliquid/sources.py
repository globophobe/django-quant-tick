"""Optional historical providers keep the native Hyperliquid market identity."""

import os
import re

from .api import normalize_coin


def coinapi_symbol(api_symbol):
    coin = normalize_coin(api_symbol)
    if not os.environ.get("COINAPI_API_KEY", "").strip():
        return None
    # HIP-3 and spot IDs use different vendor mappings and quantity contracts.
    if not re.fullmatch(r"[A-Za-z0-9]+", coin):
        return None
    return "HYPERLIQUID_PERP_" + coin.upper() + "_USDC"


def femtos_available():
    return bool(os.environ.get("FEMTOS_API_KEY", "").strip())
