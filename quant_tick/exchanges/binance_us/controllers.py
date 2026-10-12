from ..binance_format import BinanceFormatTrades, format_candles
from ..public import get_public_json


def get_binance_us_response(path, params):
    return get_public_json(
        "https://api.binance.us/api/v3",
        path,
        params,
        namespace="BINANCE_US",
        max_requests=20,
        reset_seconds=60,
    )


def binance_us_candles(api_symbol, timestamp_from, timestamp_to, resolution="1m"):
    return format_candles(
        get_binance_us_response, api_symbol, timestamp_from, timestamp_to, resolution
    )


def binance_us_trades(symbol, **kwargs):
    BinanceUSTrades(symbol, **kwargs).main()


class BinanceUSTrades(BinanceFormatTrades):
    request = staticmethod(get_binance_us_response)
