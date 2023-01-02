from datetime import datetime
from typing import Iterable, Tuple

from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_candle, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle
from ..trades import TradeData


class ConstantCandle(Candle):
    @classmethod
    def on_trades(cls, objs: Iterable[TradeData]) -> None:
        """On trades."""
        pass

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        return {
            "date": timestamp.date(),
            "thresh_attr": self.json_data["thresh_attr"],
            "thresh_value": self.json_data["thresh_attr"],
            "value": 0,
        }

    def get_cache_for_frequency(
        self, cache: dict, timestamp: datetime, frequency: Frequency
    ) -> dict:
        """Get cache for frequency."""
        next_date = timestamp.date()
        initial_cache = self.get_initial_cache(timestamp)
        # Reset cache for new era
        if frequency == Frequency.DAILY:
            if cache["date"] != next_date:
                return initial_cache
        elif frequency == Frequency.WEEKLY:
            if next_date.weekday() == 0:
                return initial_cache
        else:
            raise NotImplementedError
        return cache

    def aggregate(self, data_frame: DataFrame, cache: dict) -> Tuple[list, dict]:
        """Aggregate."""
        start = 0
        samples = []
        thresh_attr = self.json_data["thresh_attr"]
        thresh_value = self.json_data["thresh_value"]
        top_n = self.json_data["top_n"]
        for index, row in data_frame.iterrows():
            cache[thresh_attr] += row[thresh_attr]
            if cache[thresh_attr] >= thresh_value:
                df = data_frame.loc[start:index]
                sample = aggregate_candle(df, top_n)
                if "next" in cache:
                    previous = cache.pop("next")
                    sample = merge_cache(previous, sample, top_n=top_n)
                samples.append(sample)
                # Reinitialize cache
                cache[thresh_attr] = 0
                # Next index
                start = index + 1
        # Cache
        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            cache = get_next_cache(df, cache, top_n=top_n)
        return samples, cache

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
