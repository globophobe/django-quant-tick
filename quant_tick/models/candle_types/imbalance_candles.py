from datetime import datetime

import pandas as pd
from pandas import DataFrame

from quant_tick.utils import gettext_lazy as _

from .adaptive_candles import AdaptiveCandle


class ImbalanceCandle(AdaptiveCandle):
    """Imbalance candle.

    For example, 1 candle when:
    * Ticks exceed 1 standard deviation of the 7 day average tick imbalance.
    """

    def get_trade_data_summary_data_frame(self, timestamp: datetime) -> DataFrame:
        """Get trade data summary data frame."""
        trade_data_summary = self.get_trade_data_summary_for_target(timestamp)
        data_frames = []
        for t in trade_data_summary.only("file_data"):
            data_frame = t.get_data_frame()
            if data_frame is not None:
                data_frames.append(data_frame)
        if data_frames:
            return pd.concat(data_frames)
        else:
            return pd.DataFrame([])

    class Meta:
        proxy = True
        verbose_name = _("imbalance candle")
        verbose_name_plural = _("imbalance candles")
