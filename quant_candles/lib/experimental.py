import math
from itertools import tee

import numpy as np

QUANTILES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def quantile_iterator(quantiles):
    a, b = tee(quantiles)
    next(b, None)
    return zip(a, b)


def get_histogram(data_frame, quantiles=QUANTILES):
    histogram = []
    for index, quantiles in enumerate(quantile_iterator(quantiles)):
        is_last = index + 2 == len(quantiles)  # b/c iterate by 2
        lower_quantile, upper_quantile = quantiles
        lower_q = data_frame.notional.quantile(lower_quantile)
        upper_q = data_frame.notional.quantile(upper_quantile)
        lower_bound = data_frame.volume >= lower_q
        if is_last:
            upper_bound = data_frame.volume <= upper_q
        else:
            upper_bound = data_frame.volume < upper_q
        df = data_frame[lower_bound & upper_bound]
        buy_side = df[df.tickRule == 1]
        if len(df):
            value = {
                "lower": lower_quantile,
                "upper": upper_quantile,
                "volume": df.volume.sum(),
                "buyVolume": buy_side.volume.sum(),
                "notional": df.notional.sum(),
                "buyNotional": buy_side.notional.sum(),
                "ticks": len(df),
                "buyTicks": len(buy_side),
            }
            histogram.append(value)
    return histogram


def calc_volume_exponent(volume, divisor=10, decimal_places=1):
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


def calc_notional_exponent(notional, divisor=0.1, decimal_places=1):
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
