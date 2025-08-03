from django.conf import settings
from django.db import models
from django.db.models import QuerySet
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

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

    def live_trade(self, candle_data: CandleData) -> None:
        """Live."""
        raise NotImplementedError

    def on_signal(self, candle_data: CandleData, **kwargs) -> None:
        """On signal."""
        raise NotImplementedError

    def __str__(self) -> str:
        """str."""
        return self.code_name

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


class Position(models.Model):
    """Position."""

    strategy = models.ForeignKey(
        "quant_tick.Strategy",
        related_name="executions",
        on_delete=models.CASCADE,
        verbose_name=_("strategy"),
    )
    open_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        related_name="open_positions",
        verbose_name=_("candle data"),
    )
    close_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        related_name="close_positions",
        null=True,
        blank=True,
        verbose_name=_("candle data"),
    )
    json_data = JSONField(_("json data"), null=True)

    def __str__(self) -> str:
        """str."""
        code_name = self.strategy.code_name
        execution_type = self.get_execution_type_display()
        open_timestamp = self.open_candle_data.timestamp.isoformat()
        if self.close_candle_data:
            close_timestamp = self.close_candle_data.timestamp.isoformat()
            timestamp = f"{open_timestamp} - {close_timestamp}"
        else:
            timestamp = open_timestamp
        return f"{code_name} - {execution_type}: {timestamp}"

    class Meta:
        db_table = "quant_tick_position"
        verbose_name = _("position")
        verbose_name_plural = _("positions")
