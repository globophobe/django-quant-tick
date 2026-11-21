from decimal import Decimal

from quant_tick.lib import get_current_time

from .api import normalize_coin, post_hyperliquid_info


def price_context(api_symbol):
    """Sample the public mark, oracle and mid together at receipt time."""
    coin = normalize_coin(api_symbol)
    payload = {"type": "metaAndAssetCtxs"}
    if ":" in coin:
        payload["dex"] = coin.split(":", 1)[0]
    metadata, contexts = post_hyperliquid_info(payload)
    observed = get_current_time()
    universe = metadata["universe"]
    if len(universe) != len(contexts):
        raise ValueError("Hyperliquid market metadata and contexts disagree")
    matches = [
        context
        for market, context in zip(universe, contexts, strict=True)
        if market["name"] == coin
    ]
    if len(matches) != 1:
        raise ValueError("Hyperliquid omitted the requested price context")
    context = matches[0]
    prices = {}
    for name, key in (
        ("mark_price", "markPx"),
        ("oracle_price", "oraclePx"),
        ("mid_price", "midPx"),
    ):
        raw = context.get(key)
        value = Decimal(str(raw)) if raw is not None else None
        if value is None:
            if key != "midPx":
                raise ValueError("Hyperliquid omitted a reference price")
        elif not value.is_finite() or value <= 0:
            raise ValueError("Hyperliquid returned an invalid reference price")
        prices[name] = value
    return {**prices, "price_timestamp": observed, "price_block": None}
