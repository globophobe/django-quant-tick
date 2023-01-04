from datetime import datetime
from io import BytesIO
from typing import Optional, Tuple

from quant_candles.utils import gettext_lazy as _

from .constant_candles import ConstantCandle


class AdaptiveCandle(ConstantCandle):
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

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
