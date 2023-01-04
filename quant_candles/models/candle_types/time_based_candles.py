from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

from pandas import DataFrame

from quant_candles.lib import aggregate_candle, filter_by_timestamp, iter_window
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle


class TimeBasedCandle(Candle):
    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        json_data: Optional[dict] = None,
        file_data: Optional[BytesIO] = None,
    ) -> Tuple[list, Optional[dict], Optional[BytesIO]]:
        """Aggregate."""
        data = []
        window = self.json_data["window"]
        top_n = self.json_data.get("top_n", 0)
        for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, window):
            df = filter_by_timestamp(data_frame, ts_from, ts_to)
            if len(df):
                candle = aggregate_candle(df, ts_from, top_n)
                data.append(candle)
        return data, json_data, file_data

    class Meta:
        proxy = True
        verbose_name = _("time based candle")
        verbose_name_plural = _("time based candles")
