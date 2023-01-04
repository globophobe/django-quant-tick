from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_candle, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle


class ConstantCandle(Candle):
    def get_initial_cache(
        self, timestamp: datetime, **kwargs
    ) -> Tuple[Optional[dict], Optional[BytesIO]]:
        """Get initial cache."""
        return {
            "date": timestamp.date(),
            "thresh_attr": self.json_data["thresh_attr"],
            "thresh_value": self.json_data["thresh_attr"],
            "value": 0,
        }

    def get_cache(
        self,
        timestamp: datetime,
        json_data: Optional[dict] = None,
        file_data: Optional[BytesIO] = None,
    ) -> Tuple[Optional[dict], Optional[BytesIO]]:
        """Get cache."""
        json_data, file_data = super().get_cache()
        frequency = self.json_data["frequency"]
        date = timestamp.date()
        # Reset cache for new era
        if frequency == Frequency.DAILY:
            if json_data["date"] != date:
                return self.get_initial_cache(timestamp)
        elif frequency == Frequency.WEEKLY:
            if date.weekday() == 0:
                return self.get_initial_cache(timestamp)
        else:
            raise NotImplementedError
        return json_data, file_data

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        json_data: Optional[dict] = None,
        file_data: Optional[BytesIO] = None,
    ) -> Tuple[list, Optional[dict], Optional[BytesIO]]:
        """Aggregate."""
        start = 0
        data = []
        thresh_attr = self.json_data["thresh_attr"]
        thresh_value = self.json_data["thresh_value"]
        top_n = self.json_data["top_n"]
        for index, row in data_frame.iterrows():
            json_data[thresh_attr] += row[thresh_attr]
            if json_data[thresh_attr] >= thresh_value:
                df = data_frame.loc[start:index]
                candle = aggregate_candle(df, top_n)
                if "next" in json_data:
                    previous = json_data.pop("next")
                    candle = merge_cache(previous, candle, top_n=top_n)
                data.append(candle)
                # Reinitialize cache
                json_data[thresh_attr] = 0
                # Next index
                start = index + 1
        # Cache
        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            json_data = get_next_cache(df, json_data, top_n)
        return data, json_data, file_data

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
