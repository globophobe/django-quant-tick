from datetime import datetime
from typing import Optional, Tuple

from pandas import DataFrame

from quant_candles.utils import gettext_lazy as _

from ..candles import CandleCache
from .constant_candles import ConstantCandle


class AdaptiveCandle(ConstantCandle):
    def initialize(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        step: str,
        retry: bool = False,
    ) -> Tuple[datetime, datetime, Optional[dict], Optional[DataFrame]]:
        """Initialize."""
        candle_cache = (
            CandleCache.objects.filter(candle=self, timestamp__lt=timestamp_from)
            .only("timestamp", "json_data")
            .first()
        )
        if candle_cache:
            data = candle_cache.json_data
            data_frame = candle_cache.get_data_frame()
        else:
            data, data_frame = self.get_initial_cache(timestamp_from)
        return timestamp_from, timestamp_to, data, data_frame

    def get_initial_cache(
        self, timestamp: datetime
    ) -> Tuple[Optional[dict], Optional[DataFrame]]:
        """Get initial cache."""
        data = {
            "date": timestamp.date(),
            "thresh_attr": self.json_data["thresh_attr"],
            "moving_average_length": self.json_data["moving_average_length"],
            "is_ema": self.json_data["is_ema"],
        }
        return data, None

    def get_cache(
        self,
        timestamp: datetime,
        data: Optional[dict] = None,
        data_frame: Optional[DataFrame] = None,
    ) -> Tuple[Optional[dict], Optional[DataFrame]]:
        """Get cache."""
        # TODO: Fix me.
        return super().get_cache(timestamp, data, data_frame)

    def should_aggregate_candle(
        self,
        data_frame: DataFrame,
        cache_data: dict,
        cache_data_frame: Optional[DataFrame] = None,
    ) -> bool:
        """Should aggregate candle."""
        # TODO: Fix me.
        return cache_data["sample_value"] >= self.json_data["sample_value"]

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
