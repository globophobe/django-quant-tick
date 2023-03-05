from typing import Any

import mplfinance as mpf
import pandas as pd
import streamlit as st
from pandas import DataFrame
from utils import setup_django


@st.cache_data()
def get_data_frames() -> dict[Any, DataFrame]:
    """Get data frames."""
    setup_django("demo.settings.development.proxy")
    from quant_candles.constants import Exchange
    from quant_candles.lib import get_current_time, get_min_time, get_previous_time
    from quant_candles.models import Candle, Symbol

    data_frames = {}
    now = get_current_time()
    timestamp_to = get_min_time(now, value="1d")
    weekday = timestamp_to.date().weekday()

    timestamp_to = (
        timestamp_to
        if weekday == 0
        else get_previous_time(timestamp_to, value=f"{weekday + 1}d")
    )
    timestamp_from = timestamp_to - pd.Timedelta(days=7)

    symbol = Symbol.objects.get(exchange=Exchange.COINBASE, api_symbol="BTC-USD")
    candles = Candle.objects.prefetch_related("symbols").filter(symbols__in=[symbol])

    for candle in candles:
        candle_data = candle.get_data(timestamp_from, timestamp_to)
        candle_data.reverse()
        for c in candle_data:
            c.update(c.pop("json_data"))
        data_frames[candle] = pd.DataFrame(candle_data)
    return data_frames


for candle, data_frame in get_data_frames().items():
    fig, ax = mpf.plot(
        data_frame,
        title=candle.code_name,
        type="ohlc",
        show_nontrading=True,
        figsize=(15, 10),
        returnfig=True,
    )

    st.pyplot(fig)
