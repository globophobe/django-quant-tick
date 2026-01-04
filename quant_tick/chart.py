import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


def plot_renko(
    df: pd.DataFrame,
    origin_price: float = 1.0,
    target_change: float = 0.01,
    title: str = "Renko",
) -> None:
    """Plot Renko bricks colored by buy/sell imbalance with wicks.

    Args:
        df: DataFrame from RenkoBrick.get_candle_data() with columns:
            - level: brick level (y-axis position)
            - high, low: for wick calculation
            - buyVolume, volume: for color calculation
        origin_price: origin price for level boundary calculation
        target_change: target percentage change per level
        title: chart title
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    fig, ax = plt.subplots(figsize=(12, 6))
    multiplier = 1 + target_change

    for i, (_, row) in enumerate(df.iterrows()):
        level = row["level"]
        volume = float(row.get("volume", 0))
        buy_volume = float(row.get("buyVolume", 0))

        if volume > 0:
            buy_ratio = buy_volume / volume
        else:
            buy_ratio = 0.5

        color = "green" if buy_ratio > 0.5 else "red"

        # Draw brick
        ax.add_patch(
            Rectangle(
                (i, level),
                1.0,
                1.0,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
        )

        # Draw wicks (high/low normalized within level boundaries)
        level_low = origin_price * (multiplier**level)
        level_high = origin_price * (multiplier ** (level + 1))
        level_range = level_high - level_low

        x = i + 0.5
        high = float(row.get("high", level_high))
        low = float(row.get("low", level_low))

        # Upper wick (capped at level boundary)
        high_capped = min(high, level_high)
        if high_capped > level_low:
            wick_top = level + (high_capped - level_low) / level_range
            ax.vlines(x, level + 0.5, wick_top, color="black", linewidth=1)

        # Lower wick (capped at level boundary)
        low_capped = max(low, level_low)
        if low_capped < level_high:
            wick_bottom = level + (low_capped - level_low) / level_range
            ax.vlines(x, wick_bottom, level + 0.5, color="black", linewidth=1)

    level_min = df["level"].min()
    level_max = df["level"].max()

    ax.set_xlim(0, len(df))
    ax.set_ylim(level_min - 1, level_max + 2)
    ax.set_title(title)
    ax.set_xlabel("Brick #")
    ax.set_ylabel("Level")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()
