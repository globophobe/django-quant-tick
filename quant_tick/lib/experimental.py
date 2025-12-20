from decimal import Decimal


def calc_volume_exponent(
    volume: int | Decimal, divisor: int = 10, decimal_places: int = 1
) -> int:
    """Calculate volume exponent.

    Returns power of divisor if volume is a round integer (i.e. 100, 1000, 10000)
    """
    if not volume:
        return 0

    if volume % 1 != 0:
        return 0

    volume = int(volume)

    if volume % (divisor**decimal_places) == 0:
        decimal_places += 1
        while volume % (divisor**decimal_places) == 0:
            decimal_places += 1
        return decimal_places - 1

    return 0


def calc_notional_exponent(
    notional: Decimal, divisor: Decimal = Decimal("0.1"), decimal_places: int = 1
) -> int:
    """Calculate notional exponent.

    Returns power of divisor if notional is a round decimal (i.e. 1.0, 10.5, 100.0)
    """
    if not notional:
        return 0

    check_divisor = divisor * (Decimal("10") ** (decimal_places - 1))
    if notional % check_divisor == 0:
        decimal_places += 1
        while True:
            check_divisor = divisor * (Decimal("10") ** (decimal_places - 1))
            if notional % check_divisor == 0:
                decimal_places += 1
            else:
                break
        return decimal_places - 1

    return 0
