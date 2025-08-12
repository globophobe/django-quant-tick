from django.db.models import QuerySet
from pandas import DataFrame

from quant_tick.constants import Direction
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleData
from ..strategies import Signal, Strategy


class MACrossoverStrategy(Strategy):
    """MA crossover strategy."""

    def get_data_frame(self, queryset: QuerySet) -> DataFrame:
        """Get data frame."""
        data_frame = super().get_data_frame(queryset)
        moving_average_type = self.json_data["moving_average_type"]
        if moving_average_type == "sma":
            data_frame["fast_ma"] = (
                data_frame["close"].rolling(self.json_data["fast_window"]).mean()
            )
            data_frame["slow_ma"] = (
                data_frame["close"].rolling(self.json_data["slow_window"]).mean()
            )
        elif moving_average_type == "ema":
            data_frame["fast_ma"] = (
                data_frame["close"].ewm(span=self.json_data["fast_window"]).mean()
            )
            data_frame["slow_ma"] = (
                data_frame["close"].ewm(span=self.json_data["slow_window"]).mean()
            )
        return data_frame

    def backtest(self) -> None:
        """Backtest."""
        queryset = CandleData.objects.filter(candle=self.candle)
        signal = (
            Signal.objects.filter(strategy=self, close_candle_data__isnull=True)
            .select_related("open_candle_data")
            .first()
        )
        if signal:
            queryset = CandleData.objects.filter(
                timestamp__gte=signal.open_candle_data.timestamp
            )
        data_frame = self.get_data_frame(queryset)
        for index, row in enumerate(data_frame.itertuples()):
            if index >= self.json_data["slow_window"]:
                direction = (
                    Direction.LONG if row.fast_ma > row.slow_ma else Direction.SHORT
                )
                signal = self.on_data(row.obj, direction, signal)

    def live(self, candle_data: CandleData) -> None:
        """Live."""
        queryset = CandleData.objects.filter(
            candle=self.candle, timestamp__lte=candle_data.timestamp
        )
        slow_window = self.json_data["slow_window"]
        total = queryset.count()
        if total >= slow_window:
            queryset = queryset[total - slow_window :]
        data_frame = self.get_data_frame(queryset)
        if len(data_frame) >= self.json_data["slow_window"]:
            last_row = data_frame.iloc[-1]
            direction = (
                Direction.LONG
                if last_row["fast_ma"] > last_row["slow_ma"]
                else Direction.SHORT
            )
            signal = Signal.objects.filter(
                strategy=self, close_candle_data__isnull=True
            ).first()
            self.on_data(candle_data, direction, signal)

    def on_data(
        self,
        candle_data: CandleData,
        direction: Direction,
        signal: Signal | None = None,
        data: dict | None = None,
    ) -> Signal | None:
        """On data."""
        data = data or {}
        if signal:
            if signal.json_data["direction"] != direction.value:
                signal.close_candle_data = candle_data
                signal.save()
                signal = Signal.objects.create(
                    strategy=self,
                    open_candle_data=candle_data,
                    close_candle_data=None,
                    json_data={"direction": direction.value, **data},
                )
        else:
            signal = Signal.objects.create(
                strategy=self,
                open_candle_data=candle_data,
                close_candle_data=None,
                json_data={"direction": direction.value, **data},
            )
        return signal

    class Meta:
        proxy = True
        verbose_name = _("ma crossover strategy")
        verbose_name_plural = _("ma crossover strategies")
