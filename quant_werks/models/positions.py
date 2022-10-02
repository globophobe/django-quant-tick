from django.db import models

from quant_werks.utils import gettext_lazy as _

from .base import BigDecimalField


class TransactionType(models.IntegerChoices):
    BUY = 1, _("Buy")
    SELL = 2, _("Sell")
    FEE = 3, _("Fee")
    REWARD = 4, _("Reward")


class Position(models.Model):
    symbol = models.ForeignKey(
        "quant_werks.Symbol", related_name="positions", on_delete=models.CASCADE
    )
    strategy = models.ForeignKey(
        "quant_werks.Strategy",
        related_name="positions",
        null=True,
        on_delete=models.SET_NULL,
    )
    timestamp_from = models.DateTimeField(_("timestamp from"))
    timestamp_to = models.DateField(_("timestamp to"), null=True)
    realized_pnl = BigDecimalField(_("realized PnL"), null=True)
    is_paper = models.BooleanField(_("paper"), default=False)
    is_active = models.BooleanField(_("active"), default=True)

    class Meta:
        db_table = "quant_werks_position"
        verbose_name = _("position")
        verbose_name_plural = _("positions")


class Order(models.Model):
    position = models.ForeignKey(
        "quant_werks.Position", related_name="orders", on_delete=models.CASCADE
    )
    uid = models.CharField(_("uid"), max_length=255, blank=True)
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    price = BigDecimalField(_("price"))
    notional = BigDecimalField(_("notional"))
    json_data = models.JSONField(_("json data"), null=True)

    class Meta:
        db_table = "quant_werks_order"
        ordering = ("timestamp",)
        verbose_name = _("order")
        verbose_name_plural = _("orders")


class Transaction(models.Model):
    asset = models.ForeignKey(
        "quant_werks.GlobalSymbol",
        related_name="transactions",
        on_delete=models.CASCADE,
        verbose_name=_("asset"),
    )
    position = models.ForeignKey(
        "quant_werks.Position", related_name="transactions", on_delete=models.CASCADE
    )
    order = models.ForeignKey(
        "quant_werks.Order", related_name="transactions", on_delete=models.CASCADE
    )
    uid = models.CharField(_("uid"), max_length=255)
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    transaction_type = models.PositiveIntegerField(
        _("type"), choices=TransactionType.choices
    )
    price = BigDecimalField(_("price"), null=True)
    notional = BigDecimalField(_("notional"), null=True)
    json_data = models.JSONField(_("json data"), null=True)

    class Meta:
        db_table = "quant_werks_transaction"
        ordering = ("timestamp",)
        verbose_name = _("transaction")
        verbose_name_plural = _("transactions")
