"""Position sizing utilities."""


def vol_target_size(
    vol: float,
    target_vol: float,
    max_leverage: float = 1.0,
    floor: float = 1e-6,
) -> float:
    """Compute position size for volatility targeting.

    Args:
        vol: Current realized per-trade volatility
        target_vol: Target per-trade volatility
        max_leverage: Maximum allowed leverage
        floor: Minimum vol to avoid division by zero

    Returns:
        Position size multiplier
    """
    vol = max(vol, floor)
    size = target_vol / vol
    return min(size, max_leverage)


def apply_sizing(balance: float, f: float, multiplier: float) -> float:
    """Apply position sizing to balance.

    Args:
        balance: Current balance
        f: Position fraction from vol targeting
        multiplier: Trade return multiplier (e.g., 1.02 for +2%)

    Returns:
        New balance after applying sized position
    """
    trade_return = multiplier - 1.0
    sized_return = f * trade_return
    return balance * (1.0 + sized_return)
