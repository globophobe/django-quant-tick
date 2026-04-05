from decimal import Decimal

import numpy as np
from pandas import DataFrame

ZERO = Decimal("0")
DISTRIBUTION_ORIGIN = 1.0
DISTRIBUTION_STEP = 0.001
LOG_STEP = np.log1p(DISTRIBUTION_STEP)


def aggregate_distribution(df: DataFrame) -> dict:
    """Aggregate price distribution using multiplicative percent scaling."""
    if df.empty:
        return {}

    price_col = "price" if "price" in df.columns else "close"
    prices = df[price_col].values.astype(float)
    levels = np.floor(np.log(prices / DISTRIBUTION_ORIGIN) / LOG_STEP).astype(np.int64)

    df = df.copy()
    df["_level"] = levels

    distribution = {}
    has_totals = "totalVolume" in df.columns

    for level, g in df.groupby("_level"):
        is_buy = g["tickRule"] == 1

        if has_totals:
            data = {
                "ticks": int(g["totalTicks"].sum()),
                "buyTicks": (
                    int(g.loc[is_buy, "totalBuyTicks"].sum()) if is_buy.any() else 0
                ),
                "volume": g["totalVolume"].sum(),
                "buyVolume": (
                    g.loc[is_buy, "totalBuyVolume"].sum() if is_buy.any() else ZERO
                ),
                "notional": g["totalNotional"].sum(),
                "buyNotional": (
                    g.loc[is_buy, "totalBuyNotional"].sum() if is_buy.any() else ZERO
                ),
            }
        else:
            data = {
                "ticks": len(g),
                "buyTicks": int(is_buy.sum()),
                "volume": g["volume"].sum(),
                "buyVolume": g.loc[is_buy, "volume"].sum() if is_buy.any() else ZERO,
                "notional": g["notional"].sum(),
                "buyNotional": (
                    g.loc[is_buy, "notional"].sum() if is_buy.any() else ZERO
                ),
            }

        if data["notional"] > 0 or data["volume"] > 0:
            distribution[str(int(level))] = data

    return distribution


def merge_distributions(prev: dict, curr: dict) -> dict:
    """Merge two distribution dicts by summing metrics per level."""
    merged = {}
    for level in set(prev.keys()) | set(curr.keys()):
        p = prev.get(level, {})
        c = curr.get(level, {})
        merged[level] = {
            "ticks": p.get("ticks", 0) + c.get("ticks", 0),
            "buyTicks": p.get("buyTicks", 0) + c.get("buyTicks", 0),
            "volume": p.get("volume", ZERO) + c.get("volume", ZERO),
            "buyVolume": p.get("buyVolume", ZERO) + c.get("buyVolume", ZERO),
            "notional": p.get("notional", ZERO) + c.get("notional", ZERO),
            "buyNotional": p.get("buyNotional", ZERO) + c.get("buyNotional", ZERO),
        }
    return merged
