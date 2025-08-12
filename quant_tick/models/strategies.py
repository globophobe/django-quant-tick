import statistics
from decimal import Decimal

from django.conf import settings
from django.db import models
from django.db.models import QuerySet
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

from quant_tick.constants import ONE, ZERO, Direction
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField
from .candles import CandleData


def upload_file_data_to(instance: "StrategyData", filename: str) -> str:
    """Upload file data to."""
    return instance.upload_path("strategies", filename)


class BaseStrategy(models.Model):
    """Base strategy."""

    json_data = JSONField(_("json data"), null=True)
    is_active = models.BooleanField(_("is active"), default=True)

    class Meta:
        abstract = True


class Strategy(BaseStrategy, AbstractCodeName, PolymorphicModel):
    """Strategy."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
    )

    def get_data_frame(self, queryset: QuerySet) -> DataFrame:
        """Get data frame."""
        data = []
        for obj in queryset:
            is_incomplete = bool(obj.json_data.get("incomplete", False))
            data.append(
                {
                    "timestamp": obj.timestamp,
                    "obj": obj,
                    **obj.json_data,
                    **{"incomplete": is_incomplete},
                }
            )
        return DataFrame(data)

    def backtest(self) -> None:
        """Backtest."""
        raise NotImplementedError

    def live(self, candle_data: CandleData) -> None:
        """Live."""
        raise NotImplementedError

    def on_data(self, candle_data: CandleData, **kwargs) -> None:
        """On data."""
        raise NotImplementedError

    def __str__(self) -> str:
        """str."""
        return self.code_name

    @property
    def summary(self) -> dict:
        """Summary."""
        positions = list(
            self.executions.filter(close_candle_data__isnull=False).select_related(
                "open_candle_data", "close_candle_data"
            )
        )
        if len(positions):
            first_price = positions[0].open_candle_data.json_data["close"]
            last_price = positions[-1].close_candle_data.json_data["close"]
            buy_and_hold = (last_price - first_price) / first_price
        else:
            buy_and_hold = ZERO

        returns = []
        equity_curve = [ONE]
        drawdowns = []

        current_equity = ONE
        max_equity = ONE

        wins = 0
        losses = 0

        for position in positions:
            open_price = position.open_candle_data.json_data["close"]
            close_price = position.close_candle_data.json_data["close"]
            direction = position.json_data["direction"]
            if direction == Direction.LONG.value:
                r = (close_price - open_price) / open_price
            else:
                r = (open_price - close_price) / open_price

            returns.append(r)
            current_equity *= Decimal("1") + r
            equity_curve.append(current_equity)

            if current_equity > max_equity:
                max_equity = current_equity

            drawdown = (
                (max_equity - current_equity) / max_equity if max_equity > 0 else ZERO
            )
            drawdowns.append(drawdown)

            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1

        total = len(returns)

        if total > 1:
            avg_return = Decimal(str(statistics.mean([float(r) for r in returns])))
            volatility = Decimal(str(statistics.stdev([float(r) for r in returns])))
            sharpe_ratio = avg_return / volatility if volatility > 0 else ZERO
        else:
            avg_return = returns[0] if returns else ZERO
            volatility = ZERO
            sharpe_ratio = ZERO

        max_drawdown = max(drawdowns) if drawdowns else ZERO

        return {
            "equity": current_equity,
            "return": (current_equity - ONE) * 100,
            "average_return": avg_return * 100,
            "volatility": volatility * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown * 100,
            "wins": wins,
            "losses": losses,
            "win_rate": ((Decimal(wins) / Decimal(total) * 100) if total > 0 else ZERO),
            "buy_and_hold": buy_and_hold,
            "excess_return": (total - buy_and_hold) * 100,
        }

    class Meta:
        db_table = "quant_tick_strategy"
        verbose_name = _("strategy")
        verbose_name_plural = _("strategies")


class StrategyData(BaseStrategy):
    """Strategy data."""

    strategy = models.ForeignKey(
        "quant_tick.Strategy",
        related_name="strategy_data",
        on_delete=models.CASCADE,
        verbose_name=_("strategy"),
    )
    file_data = models.FileField(
        _("file data"), blank=True, upload_to=upload_file_data_to
    )

    def upload_path(self, directory: str, filename: str) -> str:
        """Upload path.

        Example:
        strategies / gentle-violin / filename
        """
        path = [f"test-{directory}"] if settings.TEST else [directory]
        path += [self.strategy.code_name, filename]
        return "/".join(path)

    class Meta:
        db_table = "quant_tick_strategy_data"
        verbose_name = verbose_name_plural = _("strategy data")


class Signal(models.Model):
    """Signal."""

    strategy = models.ForeignKey(
        "quant_tick.Strategy",
        related_name="signals",
        on_delete=models.CASCADE,
        verbose_name=_("strategy"),
    )
    open_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        related_name="+",
        verbose_name=_("candle data"),
    )
    close_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        related_name="+",
        null=True,
        blank=True,
        verbose_name=_("candle data"),
    )
    json_data = JSONField(_("json data"), null=True)

    def __str__(self) -> str:
        """str."""
        code_name = self.strategy.code_name
        open_timestamp = self.open_candle_data.timestamp.isoformat()
        if self.close_candle_data:
            close_timestamp = self.close_candle_data.timestamp.isoformat()
            timestamp = f"{open_timestamp} - {close_timestamp}"
        else:
            timestamp = open_timestamp
        return f"{code_name}: {timestamp}"

    @property
    def pnl(self) -> Decimal:
        """Profit and loss."""
        if self.close_candle_data:
            direction = self.json_data["direction"]
            open_price = self.open_candle_data.json_data["close"]
            close_price = self.close_candle_data.json_data["close"]
            if direction == Direction.LONG.value:
                return close_price - open_price
            elif direction == Direction.SHORT.value:
                return open_price - close_price
        return ZERO

    class Meta:
        db_table = "quant_tick_signal"
        ordering = ("open_candle_data__timestamp",)
        verbose_name = _("signal")
        verbose_name_plural = _("signals")
