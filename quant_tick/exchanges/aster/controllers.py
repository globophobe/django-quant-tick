"""Aster perpetual history includes the predecessor APX/Astherus market."""

from ..binance_format import BinanceFormatTrades, binance_format_candles
from ..public import get_public_json


def get_aster_response(path, params):
    return get_public_json(
        "https://fapi.asterdex.com/fapi/v3", path, params, namespace="ASTER"
    )


def aster_candles(api_symbol, timestamp_from, timestamp_to, resolution="1m"):
    return binance_format_candles(
        get_aster_response, api_symbol, timestamp_from, timestamp_to, resolution
    )


def aster_trades(symbol, **kwargs):
    AsterTrades(symbol, **kwargs).main()


class AsterTrades(BinanceFormatTrades):
    request = staticmethod(get_aster_response)
    first_trade_id = 1
