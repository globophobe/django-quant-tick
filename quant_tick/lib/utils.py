from decimal import Decimal


def safe_sum(data: list, attr: str) -> Decimal | int | None:
    """Safe sum."""
    values = [getattr(d, attr, None) for d in data]
    values = [v for v in values if v is not None]
    return sum(values) if values else None
