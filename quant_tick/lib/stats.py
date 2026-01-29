from decimal import Decimal

import numpy as np
import pandas as pd


def compute_trade_stats(
    events: pd.DataFrame,
    *,
    take_mask: pd.Series | None = None,
    return_col: str = "gross_return",
) -> dict:
    """Compute trade statistics matching chart format.

    Args:
        events: DataFrame with entry_price, exit_price, direction columns
        take_mask: Optional boolean mask for which events to include
        return_col: Column name for returns (used if available)

    Returns:
        dict with trades, wins, win_rate, pnl (as percentage)
    """
    if take_mask is None:
        take_mask = pd.Series(True, index=events.index)

    taken = events[take_mask].copy()

    valid = (
        taken["entry_price"].notna()
        & (taken["entry_price"] > 0)
        & taken["exit_price"].notna()
        & (taken["exit_price"] > 0)
        & taken["direction"].isin([1, -1])
    )
    trades = taken[valid]

    if trades.empty:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "pnl": 0.0, "max_dd": 0.0}

    # Use return_col if available, fallback to price-based calculation
    if return_col in trades.columns:
        returns = trades[return_col].astype(float)
        multipliers = 1 + returns
    else:
        # Fallback to price-based (with corrected short formula)
        entry = trades["entry_price"].astype(float)
        exit_ = trades["exit_price"].astype(float)
        long_mask = trades["direction"] == 1
        multipliers = pd.Series(index=trades.index, dtype=float)
        multipliers[long_mask] = exit_[long_mask] / entry[long_mask]
        multipliers[~long_mask] = 2 - exit_[~long_mask] / entry[~long_mask]

    # Filter non-finite multipliers
    valid_mult = np.isfinite(multipliers)
    multipliers = multipliers[valid_mult]

    if multipliers.empty:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "pnl": 0.0, "max_dd": 0.0}

    balance = Decimal("1")
    peak = Decimal("1")
    max_dd = Decimal("0")
    for m in multipliers:
        balance *= Decimal(str(m))
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd

    wins = (multipliers > 1).sum()

    return {
        "trades": len(trades),
        "wins": int(wins),
        "win_rate": float(wins / len(trades)) * 100,
        "pnl": float(balance - 1) * 100,
        "max_dd": float(max_dd) * 100,
    }


def format_stats(stats: dict) -> str:
    """Format stats like chart: Trades: N | Win: X% | PnL: Y% | MaxDD: Z%"""
    max_dd = stats.get("max_dd", 0.0)
    return (
        f"Trades: {stats['trades']} | Win: {stats['win_rate']:.0f}% | "
        f"PnL: {stats['pnl']:.1f}% | MaxDD: {max_dd:.1f}%"
    )
