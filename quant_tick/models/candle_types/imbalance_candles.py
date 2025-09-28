import math
from datetime import datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.lib import aggregate_candle, get_next_cache, merge_cache
from quant_tick.utils import gettext_lazy as _

from .constant_candles import ConstantCandle


class ImbalanceCandle(ConstantCandle):
    """Imbalance candle.

    - x_t: signed imbalance per row = (2*buy) - total (from totalBuyX/totalX for chosen sample type).
    - mu: EWMA(x_t)
    - sigma2: EWMA((x_t - mu)^2)
    - E_n: EWMA(rows per bar)
    - C: running detrended sum = Σ(x_t - mu) within the current bar
    - Close when |C| >= c * sqrt(sigma2) * sqrt(E_n), with warmup
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = super().get_initial_cache(timestamp)
        cache.update(
            {
                "mu": 0.0,  # mean of x_t
                "sigma2": 1e-12,  # variance estimate of x_t
                "E_n": self._initial_e_n,  # expected trades per bar
                "C": 0.0,  # detrended cumulative imbalance
                "n_in_bar": 0,  # rows seen in current bar
            }
        )
        return cache

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Get cache data."""
        data = super().get_cache_data(timestamp, data)
        # FIXME
        if "mu" not in data or "sigma2" not in data or "E_n" not in data:
            init = self.get_initial_cache(timestamp)
            if "next" in data:
                init["next"] = data["next"]
            if "sample_value" in data:
                init["sample_value"] = data["sample_value"]
            data = init
        return data

    @property
    def _alpha_x(self) -> float:
        """EWMA for imbalance mean."""
        return float(self.json_data.get("alpha_x", 0.001))

    @property
    def _alpha_s(self) -> float:
        """EWMA for imbalance variance."""
        return float(self.json_data.get("alpha_s", 0.001))

    @property
    def _alpha_n(self) -> float:
        """EWMA for expected trades per bar."""
        return float(self.json_data.get("alpha_n", 0.05))

    @property
    def _c(self) -> float:
        """Sensitivity multiplier for threshold."""
        return float(self.json_data.get("threshold_c", 3.0))

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
        alpha_s = self._alpha_s
        alpha_n = self._alpha_n

        for index, row in data_frame.iterrows():
            # 1) signed imbalance x_t
            x_t = float(self.get_sample_value(row))
            # 2) Update EWMAs of mean and variance of x_t
            mu_prev = float(cache_data["mu"])  # previous mean
            mu = self.ewma(mu_prev, x_t, alpha_x)
            dev = x_t - mu
            sigma2_prev = float(cache_data["sigma2"])  # previous estimate of variance
            sigma2 = self.ewma(sigma2_prev, dev * dev, alpha_s)
            # 3) Update running state for current bar
            C = float(cache_data["C"]) + dev  # detrended cumulative imbalance
            n_in_bar = int(cache_data["n_in_bar"]) + 1
            # 4) Store state back into cache
            cache_data["mu"] = mu
            cache_data["sigma2"] = sigma2
            cache_data["C"] = C
            cache_data["n_in_bar"] = n_in_bar

            # 5) Check stop condition
            if self.should_aggregate_candle(cache_data):
                df = data_frame.loc[start:index]
                candle = aggregate_candle(df)
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(previous, candle)
                data.append(candle)
                # Reinitialize cache
                E_n_prev = float(cache_data["E_n"])
                cache_data["E_n"] = self.ewma(E_n_prev, n_in_bar, alpha_n)

                cache_data["C"] = 0.0
                cache_data["n_in_bar"] = 0
                cache_data["sample_value"] = 0
                # Next index
                start = index + 1
        # Cache
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
        sigma = math.sqrt(max(cache["sigma2"], 1e-18))
        theta = self._c * sigma * math.sqrt(max(cache["E_n"], 1.0))
        warmup = max(1, min(self._min_warmup_trades, int(cache["E_n"] * 0.25)))
        if cache["n_in_bar"] < warmup:
            return False
        return abs(cache["C"]) >= theta

    class Meta:
        proxy = True
        verbose_name = _("imbalance candle")
        verbose_name_plural = _("imbalance candles")
