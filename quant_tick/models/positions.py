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
    ml_config = models.ForeignKey(
        "quant_tick.MLConfig",
        on_delete=models.CASCADE,
        verbose_name=_("ml config"),
        related_name="positions",
        null=True,
        blank=True,
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
    # LP Range Configuration
    lower_bound = models.FloatField(
        _("lower bound"),
        help_text=_("Lower bound as fraction of entry price (e.g., -0.03)"),
    )
    upper_bound = models.FloatField(
        _("upper bound"),
        help_text=_("Upper bound as fraction of entry price (e.g., 0.05)"),
    )
    borrow_ratio = models.FloatField(
        _("borrow ratio"),
        help_text=_("Asset allocation (0.5 = balanced, >0.5 = bearish, <0.5 = bullish)"),
    )
    # Timing
    entry_timestamp = models.DateTimeField(_("entry timestamp"), db_index=True)
    entry_price = BigDecimalField(_("entry price"))
    exit_timestamp = models.DateTimeField(_("exit timestamp"), null=True, blank=True)
    exit_price = BigDecimalField(_("exit price"), null=True, blank=True)
    # Outcome
    exit_reason = models.CharField(
        _("exit reason"),
        max_length=50,
        choices=ExitReason.choices,
        null=True,
        blank=True,
    )
    bars_held = models.IntegerField(
        _("bars held"),
        null=True,
        blank=True,
        help_text=_("Number of bars position was held"),
    )
    # Status & metadata
    status = models.CharField(
        _("status"),
        max_length=20,
        choices=PositionStatus.choices,
        default=PositionStatus.PENDING,
    )
    json_data = JSONField(
        _("json data"),
        null=True,
        blank=True,
        help_text=_("Additional data: p_touch_lower, p_touch_upper, width, asymmetry"),
    )

    class Meta:
        db_table = "quant_tick_position"
        verbose_name = _("position")
        verbose_name_plural = _("positions")
        ordering = ["-entry_timestamp"]
        indexes = [
            models.Index(fields=["ml_config", "position_type"]),
            models.Index(fields=["position_type", "status"]),
            models.Index(fields=["entry_timestamp"]),
        ]
