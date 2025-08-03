from pandas import DataFrame

from quant_tick.constants import Direction
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleData
from ..strategies import Position, Strategy


class MACrossoverStrategy(Strategy):
    """MA crossover strategy."""

    def get_data_frame(self, candle_data: CandleData) -> DataFrame:
        """Get data frame."""
        data_frame = super().get_data_frame(candle_data)
        moving_average_type = self.json_data["moving_average_type"]
        if moving_average_type == "sma":
            attr = "rolling"
        elif moving_average_type == "ema":
            attr = "ewm"
        else:
            raise NotImplementedError
        func = getattr(data_frame["close"], attr)
        data_frame["fast_ma"] = func(self.json_data["fast_window"]).mean()
        data_frame["slow_ma"] = func(self.json_data["slow_window"]).mean()
        return data_frame

    def backtest(self) -> None:
        """Backtest."""
        position = (
            Position.objects.filter(strategy=self, close_candle_data__isnull=True)
            .select_related("open_candle_data")
            .first()
        )
        if position:
            candle_data = position.open_candle_data
        else:
            candle_data = CandleData.objects.filter(candle=self).last()
        data_frame = self.get_data_frame(candle_data)
        for index, row in enumerate(data_frame.itertuples()):
            if index >= self.json_data["slow_window"]:
                direction = (
                    Direction.LONG if row.fast_ma > row.slow_ma else Direction.SHORT
                )
                position = self.on_signal(row.obj, direction, position)

    def on_candle_data(self, candle_data: CandleData) -> None:
        """On candle data."""
        data_frame = self.get_data_frame(candle_data)
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
