import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def renko_classic_turning_wicks(close: pd.Series, brick_size: float) -> pd.DataFrame:
    """
    Classic Renko (2-brick reversal) where *wicks appear only at swing tops/bottoms*.

    Wick rule (close-only):
    - Track max/min reached after the current last brick and before the next brick prints.
    - If trend continues, discard prior wick candidate (reset tracking).
    - If reversal is confirmed (2-brick move), attach wick to the *last brick of the prior trend*:
        * prior trend up   -> upper wick only (brick top -> run_high)
        * prior trend down -> lower wick only (run_low -> brick bottom)
    """
    close = close.dropna()
    if close.empty:
        raise ValueError("close series is empty")

    bricks = []  # dict(open, close, dir, wick_low, wick_high)
    last_close = float(close.iloc[0])
    direction = 0  # 0 unknown, +1 up, -1 down

    run_high = last_close
    run_low = last_close

    def reset_extremes(anchor: float):
        return anchor, anchor

    def finalize_turning_wick(prior_dir: int):
        nonlocal run_high, run_low
        if not bricks:
            return
        last_brick = bricks[-1]
        bottom = min(last_brick["open"], last_brick["close"])
        top = max(last_brick["open"], last_brick["close"])

        if prior_dir > 0 and run_high > top:
            last_brick["wick_high"] = run_high
        if prior_dir < 0 and run_low < bottom:
            last_brick["wick_low"] = run_low

    for price in close.iloc[1:]:
        price = float(price)
        run_high = max(run_high, price)
        run_low = min(run_low, price)

        while True:
            diff = price - last_close

            if direction == 0:
                if diff >= brick_size:
                    new_close = last_close + brick_size
                    bricks.append(
                        {
                            "open": last_close,
                            "close": new_close,
                            "dir": +1,
                            "wick_low": np.nan,
                            "wick_high": np.nan,
                        }
                    )
                    last_close = new_close
                    direction = +1
                    run_high, run_low = reset_extremes(last_close)
                    continue
                elif diff <= -brick_size:
                    new_close = last_close - brick_size
                    bricks.append(
                        {
                            "open": last_close,
                            "close": new_close,
                            "dir": -1,
                            "wick_low": np.nan,
                            "wick_high": np.nan,
                        }
                    )
                    last_close = new_close
                    direction = -1
                    run_high, run_low = reset_extremes(last_close)
                    continue
                else:
                    break

            elif direction == +1:
                if diff >= brick_size:
                    # trend continues up: reset extremes (discard wick candidate)
                    new_close = last_close + brick_size
                    bricks.append(
                        {
                            "open": last_close,
                            "close": new_close,
                            "dir": +1,
                            "wick_low": np.nan,
                            "wick_high": np.nan,
                        }
                    )
                    last_close = new_close
                    run_high, run_low = reset_extremes(last_close)
                    continue
                elif diff <= -2 * brick_size:
                    # reversal confirmed: wick belongs to swing top (last up brick)
                    finalize_turning_wick(prior_dir=+1)

                    o = last_close - brick_size
                    c = last_close - 2 * brick_size
                    bricks.append(
                        {
                            "open": o,
                            "close": c,
                            "dir": -1,
                            "wick_low": np.nan,
                            "wick_high": np.nan,
                        }
                    )
                    last_close = c
                    direction = -1
                    run_high, run_low = reset_extremes(last_close)
                    continue
                else:
                    break

            else:  # direction == -1
                if diff <= -brick_size:
                    # trend continues down: reset extremes (discard wick candidate)
                    new_close = last_close - brick_size
                    bricks.append(
                        {
                            "open": last_close,
                            "close": new_close,
                            "dir": -1,
                            "wick_low": np.nan,
                            "wick_high": np.nan,
                        }
                    )
                    last_close = new_close
                    run_high, run_low = reset_extremes(last_close)
                    continue
                elif diff >= 2 * brick_size:
                    # reversal confirmed: wick belongs to swing bottom (last down brick)
                    finalize_turning_wick(prior_dir=-1)

                    o = last_close + brick_size
                    c = last_close + 2 * brick_size
                    bricks.append(
                        {
                            "open": o,
                            "close": c,
                            "dir": +1,
                            "wick_low": np.nan,
                            "wick_high": np.nan,
                        }
                    )
                    last_close = c
                    direction = +1
                    run_high, run_low = reset_extremes(last_close)
                    continue
                else:
                    break

    return pd.DataFrame(bricks)


def plot_renko_turning_wicks(
    bricks: pd.DataFrame, brick_size: float, title="Renko (turning-point wicks only)"
):
    if bricks.empty:
        raise ValueError("No bricks created.")

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, row in bricks.reset_index(drop=True).iterrows():
        o, c, d = float(row.open), float(row.close), int(row.dir)
        bottom, top = min(o, c), max(o, c)

        face = "green" if d > 0 else "red"
        ax.add_patch(
            Rectangle(
                (i, bottom),
                1.0,
                top - bottom,
                facecolor=face,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        x = i + 0.5
        # Only draw wick if present, and only in the brick's direction
        if d > 0 and pd.notna(row.wick_high) and float(row.wick_high) > top:
            ax.vlines(x, top, float(row.wick_high), color="black", linewidth=1.0)
        if d < 0 and pd.notna(row.wick_low) and float(row.wick_low) < bottom:
            ax.vlines(x, float(row.wick_low), bottom, color="black", linewidth=1.0)

    ymin = bricks[["open", "close"]].min().min()
    ymax = bricks[["open", "close"]].max().max()
    ymin = min(
        ymin, pd.to_numeric(bricks["wick_low"], errors="coerce").min(skipna=True)
    )
    ymax = max(
        ymax, pd.to_numeric(bricks["wick_high"], errors="coerce").max(skipna=True)
    )

    ax.set_xlim(0, len(bricks))
    ax.set_ylim(ymin - brick_size, ymax + brick_size)
    ax.set_title(title)
    ax.set_xlabel("Brick #")
    ax.set_ylabel("Price")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()


# Demo
np.random.seed(7)
close = pd.Series(100 + np.cumsum(np.random.normal(0, 0.9, size=450)))

brick_size = 1.0
bricks = renko_classic_turning_wicks(close, brick_size)
plot_renko_turning_wicks(
    bricks, brick_size, title=f"Renko turning wicks only (brick={brick_size})"
)
