import math
from decimal import Decimal

import numpy as np


def get_top_n(data_frame: DataFrame, sample_type: SampleType, top_n: int) -> list[dict]:
    """Get top N"""
    index = (
        data_frame[data_frame.volume > 0][sample_type]
        .astype(float)
        .nlargest(top_n)
        .index
    )
    return data_frame[data_frame.index.isin(index)]


def calc_volume_exponent(
    volume: int, divisor: int = 10, decimal_places: int = 1
) -> int:
    """Calculate volume exponent."""
    if volume > 0:
        is_round = volume % math.pow(divisor, decimal_places) == 0
        if is_round:
            decimal_places += 1
            stop_execution = False
            while not stop_execution:
                is_round = volume % math.pow(divisor, decimal_places) == 0
                if is_round:
                    decimal_places += 1
                else:
                    stop_execution = True
            return decimal_places - 1
        else:
            return 0
    else:
        # WTF Bybit!
        return 0


def calc_notional_exponent(
    notional: Decimal, divisor: float = 0.1, decimal_places: int = 1
) -> int:
    """Calculate notional exponent."""
    if notional > 0:
        # Not scientific notation, max 10 decimal places
        decimal = format(notional, f".{10}f").lstrip().rstrip("0")
        # Only mantissa, plus trailing zero
        value = str(decimal).split(".")[1] + "0"
        for i in range(len(value)):
            val = float(f"0.{value[i:]}")
            is_close = np.isclose(val, 0)
            if not is_close:
                decimal_places += 1
            else:
                return -decimal_places + 1
        else:
            return 0
    else:
        # WTF Bybit!
        return 0
