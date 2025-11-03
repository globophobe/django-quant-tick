from django.db import models

from quant_tick.constants import ExitReason, PositionStatus, PositionType
from quant_tick.utils import gettext_lazy as _

from .base import BigDecimalField, JSONField


class Position(models.Model):
    """Position."""

    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        verbose_name=_("symbol"),
        related_name="positions",
    )
    ml_run = models.ForeignKey(
        "quant_tick.MLRun",
        on_delete=models.CASCADE,
        verbose_name=_("ml run"),
        related_name="positions",
    )
    ml_signal = models.ForeignKey(
        "quant_tick.MLSignal",
        on_delete=models.SET_NULL,
        verbose_name=_("ml signal"),
        related_name="positions",
        null=True,
        blank=True,
    )
    position_type = models.CharField(
        _("position type"),
        max_length=20,
        choices=PositionType.choices,
        default=PositionType.BACKTEST,
    )
    entry_timestamp = models.DateTimeField(_("entry timestamp"), db_index=True)
    entry_price = BigDecimalField(_("entry price"))
    take_profit = BigDecimalField(_("take profit"), null=True, blank=True)
    stop_loss = BigDecimalField(_("stop loss"), null=True, blank=True)
    max_duration = models.DateTimeField(_("max duration"), null=True, blank=True)
    exit_timestamp = models.DateTimeField(_("exit timestamp"), null=True, blank=True)
    exit_price = BigDecimalField(_("exit price"), null=True, blank=True)
    exit_reason = models.CharField(
        _("exit reason"),
        max_length=50,
        choices=ExitReason.choices,
        null=True,
        blank=True,
    )
    side = models.SmallIntegerField(_("side"))
    size = BigDecimalField(_("size"))
    fees = BigDecimalField(_("fees"), null=True, blank=True)
    status = models.CharField(
        _("status"),
        max_length=20,
        choices=PositionStatus.choices,
        default=PositionStatus.PENDING,
    )
    json_data = JSONField(_("json data"), null=True, blank=True)
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)
    updated_at = models.DateTimeField(_("updated at"), auto_now=True)

    class Meta:
        db_table = "quant_tick_position"
        verbose_name = _("position")
        verbose_name_plural = _("positions")
        ordering = ["-entry_timestamp"]
        indexes = [
            models.Index(fields=["ml_run", "position_type"]),
            models.Index(fields=["position_type", "status"]),
            models.Index(fields=["entry_timestamp"]),
        ]
