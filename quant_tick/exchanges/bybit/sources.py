"""Optional trade-history mappings for Bybit spot and USDT perpetuals."""

import os
import re

from quant_tick.constants import Exchange


def coinapi_symbol(symbol):
    if not os.environ.get("COINAPI_API_KEY", "").strip():
        return None
    market = {Exchange.BYBIT: "SPOT", Exchange.BYBIT_LINEAR: "PERP"}.get(
        symbol.exchange
    )
    if market is None:
        return None
    # USDC perpetuals and inverse contracts have different symbol/unit contracts.
    pair = re.fullmatch(r"([A-Z0-9]+)(USDT)", symbol.api_symbol)
    if pair is None:
        return None
    base, quote = pair.groups()
    return f"BYBIT_{market}_{base}_{quote}"
