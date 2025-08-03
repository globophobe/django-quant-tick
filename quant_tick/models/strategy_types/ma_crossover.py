from django.db.models import QuerySet
from pandas import DataFrame

from quant_tick.constants import Direction
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleData
from ..strategies import Position, Strategy


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
        position = (
            Position.objects.filter(strategy=self, close_candle_data__isnull=True)
            .select_related("open_candle_data")
            .first()
        )
        if position:
            queryset = CandleData.objects.filter(
                timestamp__gte=position.open_candle_data.timestamp
            )
        data_frame = self.get_data_frame(queryset)
        for index, row in enumerate(data_frame.itertuples()):
            if index >= self.json_data["slow_window"]:
                direction = (
                    Direction.LONG if row.fast_ma > row.slow_ma else Direction.SHORT
                )
                position = self.on_signal(row.obj, direction, position)

    def live_trade(self, candle_data: CandleData) -> None:
        """Live trade."""
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
            position = Position.objects.filter(
                strategy=self, close_candle_data__isnull=True
            ).first()
            self.on_signal(candle_data, direction, position)

    def on_signal(
        self,
        candle_data: CandleData,
        direction: Direction,
        position: Position | None = None,
        data: dict | None = None,
    ) -> Position | None:
        """On signal."""
        data = data or {}
        if position:
            if position.json_data["direction"] != direction.value:
                position.close_candle_data = candle_data
                position.save()
                position = Position.objects.create(
                    strategy=self,
                    open_candle_data=candle_data,
                    close_candle_data=None,
                    json_data={"direction": direction.value, **data},
                )
        else:
            position = Position.objects.create(
                strategy=self,
                open_candle_data=candle_data,
                close_candle_data=None,
                json_data={"direction": direction.value, **data},
            )
        return position

    class Meta:
        proxy = True
        verbose_name = _("ma crossover strategy")
        verbose_name_plural = _("ma crossover strategies")
