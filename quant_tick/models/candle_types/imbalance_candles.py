from datetime import datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.lib import aggregate_candle, get_next_cache, merge_cache
from quant_tick.utils import gettext_lazy as _

from .constant_candles import ConstantCandle


class ImbalanceCandle(ConstantCandle):
    """Imbalance candle.

    - x_t: signed imbalance per row = (2*buy) - total
    - E_x: EWMA(|x_t|)
    - E_n: EWMA(rows per bar)
    - theta: E_n * E_x
    - Close when |Σx_t| >= theta, with warmup
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = super().get_initial_cache(timestamp)
        cache.update(
            {
                "E_x": 0.0,
                "E_n": self._initial_e_n,
                "theta": 0.0,
                "n_in_bar": 0,
            }
        )
        return cache

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Get cache data."""
        data = super().get_cache_data(timestamp, data)
        if "E_x" not in data or "E_n" not in data:
            init = self.get_initial_cache(timestamp)
            if "next" in data:
                init["next"] = data["next"]
            if "sample_value" in data:
                init["sample_value"] = data["sample_value"]
            data = init
        return data

    @property
    def _alpha_x(self) -> float:
        """EWMA for expected imbalance."""
        return float(self.json_data.get("alpha_x", 0.05))

    @property
    def _alpha_n(self) -> float:
        """EWMA for expected trades per bar."""
        return float(self.json_data.get("alpha_n", 0.05))

    @property
    def _initial_e_n(self) -> float:
        """Expected trades per bar."""
        return float(self.json_data.get("initial_expected_trades_per_bar", 200.0))

    @property
    def _min_warmup_trades(self) -> int:
        """Minimum warmup trades."""
        return int(self.json_data.get("min_warmup_trades", 100))

    def ewma(self, prev: float, obs: float, alpha: float) -> float:
        """EWMA."""
        return (alpha * obs) + ((1.0 - alpha) * prev)

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate."""
        start = 0
        data: list[dict] = []
        alpha_x = self._alpha_x
        alpha_n = self._alpha_n

        for index, row in data_frame.iterrows():
            x_t = float(self.get_sample_value(row))
            E_x_prev = float(cache_data["E_x"])
            E_x = self.ewma(E_x_prev, abs(x_t), alpha_x)
            theta = float(cache_data["theta"]) + x_t
            n_in_bar = int(cache_data["n_in_bar"]) + 1

            cache_data["E_x"] = E_x
            cache_data["theta"] = theta
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
                cache_data["theta"] = 0.0
                cache_data["n_in_bar"] = 0
                cache_data["sample_value"] = 0
                start = index + 1

        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            cache_data = get_next_cache(df, cache_data)

        data, cache_data = self.get_incomplete_candle(timestamp_to, data, cache_data)
        return data, cache_data

    def get_sample_value(self, row: tuple) -> Decimal | int:
        """Get sample value."""
        sample_type = self.json_data["sample_type"]
        s_type = sample_type.title()
        buy = row[f"totalBuy{s_type}"]
        total = row[f"total{s_type}"]
        return (2 * buy) - total

    def should_aggregate_candle(self, cache: dict) -> bool:
        """Should aggregate candle."""
        threshold = cache["E_n"] * cache["E_x"]
        warmup = max(1, min(self._min_warmup_trades, int(cache["E_n"] * 0.25)))
        if cache["n_in_bar"] < warmup:
            return False
        return abs(cache["theta"]) >= threshold

    class Meta:
        proxy = True
        verbose_name = _("imbalance candle")
        verbose_name_plural = _("imbalance candles")
