"""Authenticated public market data; CoinAPI IDs keep vendor and venue feeds separate."""

import os
import re

from django.core.exceptions import ImproperlyConfigured

from ..public import get_public_json


def require_spot_symbol(api_symbol):
    if not re.fullmatch(r"[A-Z0-9_-]+_SPOT_[A-Z0-9]+_[A-Z0-9]+", api_symbol):
        raise ValueError(
            "CoinAPI requires a spot symbol ID, e.g. BINANCE_SPOT_BTC_USDT"
        )


def get_coinapi_response(path, params):
    api_key = os.environ.get("COINAPI_API_KEY", "").strip()
    if not api_key:
        raise ImproperlyConfigured("Set COINAPI_API_KEY to collect CoinAPI market data")
    payload = get_public_json(
        "https://rest.coinapi.io",
        path,
        params,
        namespace="COINAPI",
        headers={"X-CoinAPI-Key": api_key},
        retry=3,
    )
    if not isinstance(payload, list):
        raise TypeError("CoinAPI returned an unexpected market data response")
    return payload
