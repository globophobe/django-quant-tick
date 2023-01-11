from datetime import datetime
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
        cache_data: Optional[dict] = None,
        cache_data_frame: Optional[DataFrame] = None,
    ) -> Tuple[list, Optional[dict]]:
        """Aggregate."""
        data = []
        window = self.json_data["window"]
        for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, window):
            df = filter_by_timestamp(data_frame, ts_from, ts_to)
            if len(df):
                candle = self.aggregate_candle(ts_from, df)
                data.append(candle)
        return data, cache_data

    def aggregate_candle(self, timestamp: datetime, data_frame: DataFrame) -> dict:
        """Aggregate candle."""
        return aggregate_candle(
            data_frame, timestamp, top_n=self.json_data.get("top_n", 0)
        )

    class Meta:
        proxy = True
        verbose_name = _("time based candle")
        verbose_name_plural = _("time based candles")
