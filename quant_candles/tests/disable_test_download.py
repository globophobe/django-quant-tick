import datetime

import pandas as pd

from quant_candles.exchanges.bitmex.constants import XBTUSD
from quant_candles.exchanges.bitmex.controllers import BitmexTradesS3
from quant_candles.exchanges.bybit.constants import BTCUSD
from quant_candles.exchanges.bybit.controllers import BybitTradesS3
from quant_candles.lib import get_current_time, gzip_downloader


def assert_200(controller, date):
    url = controller.get_url(date)
    data_frame = gzip_downloader(url)
    assert len(data_frame) > 0


def assert_404(controller):
    now = get_current_time()
    delta = now + pd.Timedelta("1d")
    tomorrow = delta.date()
    url = controller.get_url(tomorrow)
    data_frame = gzip_downloader(url)
    assert data_frame is None


def test_bitmex_200():
    controller = BitmexTradesS3(XBTUSD)
    date = datetime.date(2016, 5, 14)
    assert_200(controller, date)


def test_bitmex_404():
    controller = BitmexTradesS3(XBTUSD)
    assert_404(controller)


def test_bybit_200():
    controller = BybitTradesS3(BTCUSD)
    date = datetime.date(2020, 1, 1)
    assert_200(controller, date)


def test_bybit_404():
    controller = BybitTradesS3(BTCUSD)
    assert_404(controller)
