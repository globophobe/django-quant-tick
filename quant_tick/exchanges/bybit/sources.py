"""Optional trade-history mappings for Bybit spot and USDT perpetuals."""

import os
import re

from quant_tick.constants import Exchange


def coinapi_symbol(symbol):
    if not os.environ.get("COINAPI_API_KEY", "").strip():
        return None
    prefix = {
        Exchange.BYBIT: "BYBITSPOT_SPOT",
        Exchange.BYBIT_LINEAR: "BYBIT_PERP",
    }.get(symbol.exchange)
    if prefix is None:
        return None
    # USDC perpetuals and inverse contracts have different symbol/unit contracts.
    pair = re.fullmatch(r"([A-Z0-9]+)(USDT)", symbol.api_symbol)
    if pair is None:
        return None
    base, quote = pair.groups()
    return f"{prefix}_{base}_{quote}"
