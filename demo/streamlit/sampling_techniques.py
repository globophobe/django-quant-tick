from typing import Any

import mplfinance as mpf
import pandas as pd
import streamlit as st
from pandas import DataFrame
from utils import setup_django

setup_django("demo.settings.development.proxy")


@st.cache_data()
def get_data_frames() -> dict[Any, DataFrame]:
    """Get data frames."""
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
        data_frame = pd.DataFrame(candle_data).set_index("timestamp")
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "buyVolume",
            "notional",
            "buyNotional",
        ):
            data_frame[column] = data_frame[column].astype(float)
        data_frames[candle] = data_frame
    return data_frames


for candle, data_frame in get_data_frames().items():
    from quant_candles.constants import Frequency
    from quant_candles.models import AdaptiveCandle, TimeBasedCandle

    total_candles = len(data_frame)
    if isinstance(candle, TimeBasedCandle):
        window = candle.json_data["window"]
        delta = pd.Timedelta(window)
        total_minutes = int(delta.total_seconds() / Frequency.HOUR)
        title = f"{total_minutes} minutes, {total_candles} candles"
    elif isinstance(candle, AdaptiveCandle):
        sample_type = candle.json_data["sample_type"].capitalize()
        title = f"{sample_type}, {total_candles} candles"
    fig, ax = mpf.plot(
        data_frame,
        title=title,
        type="candle",
        show_nontrading=True,
        figsize=(15, 10),
        style="nightclouds",
        returnfig=True,
    )

    st.pyplot(fig)
