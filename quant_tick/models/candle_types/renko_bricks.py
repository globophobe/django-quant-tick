from datetime import datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.lib import aggregate_candle, get_next_cache, merge_cache
from quant_tick.utils import gettext_lazy as _

from .constant_candles import ConstantCandle


class RenkoBrick(ConstantCandle):
    """Renko brick.

    Using multiplicative percent scaling, with relative normalization.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        return {
            "normalization_factor": None,
            "level": None,
            "direction": None,
        }

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate."""
        start = 0
        data = []
        for index, row in data_frame.iterrows():
            if self.should_aggregate_candle(row.price, cache_data):
                df = data_frame.loc[start:index]
                candle = aggregate_candle(df)
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(previous, candle)
                data.append(candle)
                # Next index
                start = index + 1
        # Cache
        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            cache_data = get_next_cache(df, cache_data)
        data, cache_data = self.get_incomplete_candle(timestamp_to, data, cache_data)
        return data, cache_data

    def should_aggregate_candle(self, price: Decimal, data: dict) -> bool:
        """Should aggregate candle."""
        target_change = self.json_data["target_percentage_change"]
        factor = data.get("normalization_factor")
        if factor is None:
            factor = Decimal(1) / price
            data["normalization_factor"] = factor
            data["level"] = None
            data["direction"] = None
            return True
        else:
            normalized_price = price * factor
            current_level = int((normalized_price - Decimal(1)) / target_change)
            direction = data.get("direction")
            if direction is None:
                data["level"] = current_level
                data["direction"] = 1
                return True
            else:
                last_level = data["level"]
                change = current_level - last_level
                if direction == 1:
                    if change >= 1:
                        data["level"] = last_level + 1
                        return True
                    elif change <= -2:
                        data["level"] = last_level - 1
                        data["direction"] = -1
                        return True
                elif direction == -1:
                    if change <= -1:
                        data["level"] = last_level - 1
                        return True
                    elif change >= 2:
                        data["level"] = last_level + 1
                        data["direction"] = 1
                        return True
        return False

    class Meta:
        proxy = True
        verbose_name = _("renko brick")
        verbose_name_plural = _("renko bricks")
