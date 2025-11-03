from datetime import datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.lib import aggregate_candle, get_next_cache, merge_cache
from quant_tick.utils import gettext_lazy as _

from .imbalance_candles import ImbalanceCandle


class RunCandle(ImbalanceCandle):
    """Run candle.

    - θ_t: buy proportion per row = totalBuyX / totalX
    - E_θ: EWMA(θ_t)
    - E_n: EWMA(rows per bar)
    - buy_run, sell_run: consecutive buy/sell sequence lengths
    - max_run: max(buy_run, sell_run)
    - Close when max_run >= E_n * E_θ, with warmup
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = super().get_initial_cache(timestamp)
        cache.update(
            {
                "buy_run": 0,
                "sell_run": 0,
            }
        )
        return cache

    def get_sample_value(self, row: tuple) -> Decimal | int:
        """Get sample value."""
        sample_type = self.json_data["sample_type"]
        s_type = sample_type.title()
        buy = row[f"totalBuy{s_type}"]
        total = row[f"total{s_type}"]
        if total == 0:
            return 0
        return buy / total

    def aggregate(self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame, cache_data: dict) -> None:
        """Aggregate."""
        start = 0
        data: list[dict] = []
        alpha_x = self._alpha_x
        alpha_n = self._alpha_n

        for index, row in data_frame.iterrows():
            theta_t = float(self.get_sample_value(row))
            E_theta_prev = float(cache_data["E_x"])
            E_theta = self.ewma(E_theta_prev, theta_t, alpha_x)
            n_in_bar = int(cache_data["n_in_bar"]) + 1

            if theta_t > 0.5:
                cache_data["buy_run"] = cache_data.get("buy_run", 0) + 1
                cache_data["sell_run"] = 0
            else:
                cache_data["sell_run"] = cache_data.get("sell_run", 0) + 1
                cache_data["buy_run"] = 0

            cache_data["E_x"] = E_theta
            cache_data["n_in_bar"] = n_in_bar

            if self.should_aggregate_candle(cache_data):
                df = data_frame.loc[start:index]
                candle = aggregate_candle(df)
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(previous, candle)
                data.append(candle)
                E_n_prev = float(cache_data["E_n"])
                cache_data["E_n"] = self.ewma(E_n_prev, n_in_bar, alpha_n)
                cache_data["n_in_bar"] = 0
                cache_data["sample_value"] = 0
                cache_data["buy_run"] = 0
                cache_data["sell_run"] = 0
                start = index + 1

        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            cache_data = get_next_cache(df, cache_data)

        data, cache_data = self.get_incomplete_candle(timestamp_to, data, cache_data)
        return data, cache_data

    def should_aggregate_candle(self, cache: dict) -> bool:
        """Should aggregate candle."""
        threshold = cache["E_n"] * cache["E_x"]
        warmup = max(1, min(self._min_warmup_trades, int(cache["E_n"] * 0.25)))
        if cache["n_in_bar"] < warmup:
            return False
        max_run = max(cache["buy_run"], cache["sell_run"])
        return max_run >= threshold

    class Meta:
        proxy = True
        verbose_name = _("run candle")
        verbose_name_plural = _("run candles")
