from typing import Iterable, Tuple

from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_candle, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle
from ..trades import TradeData


class TimeBasedCandle(Candle):
    @classmethod
    def on_trades(cls, objs: Iterable[TradeData]) -> None:
        """On trades."""
        pass

    def get_range(self, time_delta) -> Tuple[str, int]:
        """Get range."""
        total_seconds = time_delta.total_seconds()
        one_minute = 60.0
        one_hour = 3600.0
        if total_seconds < one_minute:
            unit = "seconds"
            step = total_seconds
        elif total_seconds >= one_minute and total_seconds <= one_hour:
            unit = "minutes"
            step = total_seconds / one_minute
        elif total_seconds > one_hour:
            unit = "hours"
            step = total_seconds / one_hour
        else:
            raise NotImplementedError
        step = int(step)
        assert total_seconds <= one_hour, f"{step} {unit} not supported"
        assert 60 % step == 0, f"{step} not divisible by 60"
        return unit, step

    def aggregate_candles(
        data_frame,
        timestamp_from,
        timestamp_to,
        time_delta,
        top_n=0,
        cache={},
        as_dict=False,
        return_cache=True,
    ):
        samples = []
        period = pendulum.period(timestamp_from, timestamp_to)
        unit, step = get_range(time_delta)
        for index, start in enumerate(period.range(unit, step)):
            end = start + pd.Timedelta(f"{step}{unit}")
            df = data_frame[
                (data_frame.timestamp >= start) & (data_frame.timestamp < end)
            ]
            # Maybe trades
            if len(df):
                if "symbol" in df.columns:
                    symbols = df.symbol.unique()
                    assert len(symbols) == 1
                    open_price = cache[symbols[0]]
                else:
                    open_price = cache["open"]
                sample = aggregate_rows(
                    df,
                    # Open timestamp, or won't be in partition
                    timestamp=start,
                    open_price=open_price,
                    top_n=top_n,
                )
                cache["open"] = sample["close"]
                samples.append(sample)
        data = pd.DataFrame(samples) if not top_n and not as_dict else samples
        return data, cache if return_cache else data

    class Meta:
        proxy = True
        verbose_name = _("time based candle")
        verbose_name_plural = _("time based candles")
