from django.contrib import admin
from django.contrib.auth.models import Group, User
from django.http import HttpRequest
from semantic_admin import SemanticModelAdmin

from quant_tick.models import Candle, Symbol, TaskState

admin.site.unregister(User)
admin.site.unregister(Group)


@admin.register(Symbol)
class SymbolAdmin(SemanticModelAdmin):
    """Symbol admin."""

    list_display = ("__str__",)
    fields = (
        ("exchange", "api_symbol"),
        ("aggregate_trades", "significant_trade_filter"),
    )


@admin.register(Candle)
class CandleAdmin(SemanticModelAdmin):
    """Candle admin."""

    list_display = ("code_name", "symbol", "is_active")
    list_filter = ("is_active", "symbol")


@admin.register(TaskState)
class TaskStateAdmin(SemanticModelAdmin):
    """Task state admin."""

    list_display = (
        "name",
        "exchange",
        "locked_until",
        "next_fetch_at",
        "recent_error_count",
        "recent_error_at",
    )
    list_filter = ("name", "exchange")
    readonly_fields = (
        "name",
        "exchange",
        "locked_until",
        "next_fetch_at",
        "recent_error_count",
        "recent_error_at",
    )
    search_fields = ("name", "exchange")

    def has_add_permission(self, request: HttpRequest) -> bool:
        """Task states created by app workflows."""
        return False
