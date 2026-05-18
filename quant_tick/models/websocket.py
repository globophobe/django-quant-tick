from typing import Any

from django.db import models
from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.constants import Exchange
from quant_tick.lib import to_decimal_or_none, to_utc_datetime

from .base import JSONField
from .symbols import Symbol

DECIMAL_FIELDS = (
    "price",
    "volume",
    "notional",
    "high",
    "low",
    "totalBuyVolume",
    "totalVolume",
    "totalBuyNotional",
    "totalNotional",
)
INT_FIELDS = (
    "nanoseconds",
    "tickRule",
    "ticks",
    "totalBuyTicks",
    "totalTicks",
)


def to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def parse_trade_row(row: dict) -> dict:
    data = dict(row)
    data["timestamp"] = to_utc_datetime(data["timestamp"])
    for field in DECIMAL_FIELDS:
        if field in data:
            data[field] = to_decimal_or_none(data[field])
    for field in INT_FIELDS:
        if field in data:
            data[field] = to_int_or_none(data[field])
    return data


class WebSocketDataQuerySet(QuerySet):
    def for_symbol(self, symbol: Symbol) -> QuerySet:
        return self.filter(
            exchange=symbol.exchange,
            api_symbol=symbol.api_symbol,
            significant_trade_filter=symbol.significant_trade_filter or 0,
        )


class WebSocketData(models.Model):
    """Trade data from websockets."""

    exchange = models.CharField(_("exchange"), choices=Exchange.choices, max_length=255)
    api_symbol = models.CharField(_("API symbol"), max_length=255)
    significant_trade_filter = models.PositiveIntegerField(
        _("significant trade filter"),
        default=0,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    raw_trades = JSONField(_("raw trades"), default=list, blank=True)
    aggregated_trades = JSONField(_("aggregated trades"), default=list, blank=True)
    filtered_trades = JSONField(_("filtered trades"), default=list, blank=True)

    objects = WebSocketDataQuerySet.as_manager()

    @staticmethod
    def get_trades(value: list[dict]) -> list[dict]:
        trades = value or []
        if not isinstance(trades, list):
            raise ValueError("trades must be a list.")
        for trade in trades:
            if not isinstance(trade, dict):
                raise ValueError("trades items must be objects.")
        return trades

    @classmethod
    def get_data_frame(cls, value: list[dict]) -> DataFrame | None:
        trades = cls.get_trades(value)
        if not trades:
            return None
        data = DataFrame(parse_trade_row(row) for row in trades)
        return data.reset_index(drop=True)

    def get_data_frames(
        self,
        symbol: Symbol,
    ) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None]:
        raw_trades = self.get_data_frame(self.raw_trades)
        aggregated_trades = self.get_data_frame(self.aggregated_trades)
        filtered_trades = self.get_data_frame(self.filtered_trades)
        if not symbol.save_raw:
            raw_trades = None
        if not symbol.save_aggregated:
            aggregated_trades = None
        if not symbol.significant_trade_filter:
            filtered_trades = None
        return raw_trades, aggregated_trades, filtered_trades

    class Meta:
        db_table = "quant_tick_websocket_data"
        ordering = ("timestamp", "exchange", "api_symbol")
        verbose_name = verbose_name_plural = _("websocket data")
        constraints = [
            models.UniqueConstraint(
                fields=(
                    "exchange",
                    "api_symbol",
                    "significant_trade_filter",
                    "timestamp",
                ),
                name="quant_tick_websocket_data_unique",
            ),
        ]
