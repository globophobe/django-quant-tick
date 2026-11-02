from decimal import Decimal

from ..public import get_public_json


def get_okx_response(path, params):
    payload = get_public_json(
        "https://www.okx.com/api/v5", path, params, namespace="OKX"
    )
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX {path}: {payload}")
    return payload["data"]


def contract_value(api_symbol):
    """Return base quantity per contract for linear swaps; spot size is already base."""
    if not api_symbol.endswith("-SWAP"):
        return Decimal(1)
    rows = get_okx_response(
        "/public/instruments", {"instType": "SWAP", "instId": api_symbol}
    )
    if len(rows) != 1:
        raise ValueError(f"OKX instrument is unavailable: {api_symbol}")
    instrument = rows[0]
    if (
        instrument["instId"] != api_symbol
        or instrument["ctType"] != "linear"
        or instrument["ctValCcy"] != api_symbol.split("-")[0]
    ):
        raise ValueError("OKX swaps require a linear base-denominated contract")
    value = Decimal(instrument["ctVal"])
    if not value.is_finite() or value <= 0:
        raise ValueError("OKX contract value must be positive")
    return value
