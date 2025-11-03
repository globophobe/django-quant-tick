from django.contrib import admin
from django.contrib.auth.models import Group, User
from semantic_admin import SemanticModelAdmin, SemanticTabularInline

from quant_tick.models import GlobalSymbol, Symbol

admin.site.unregister(User)
admin.site.unregister(Group)


class SymbolInline(SemanticTabularInline):
    """Symbol inline."""

    model = Symbol
    extra = 0


@admin.register(GlobalSymbol)
class GlobalSymbolAdmin(SemanticModelAdmin):
    """Global symbol admin."""

    list_display = ("__str__",)
    fields = ("name",)
    inlines = (SymbolInline,)


@admin.register(Symbol)
class SymbolAdmin(SemanticModelAdmin):
    """Symbol admin."""

    list_display = ("__str__",)
    fields = (
        ("global_symbol", "exchange"),
        ("symbol_type", "api_symbol"),
        ("aggregate_trades", "significant_trade_filter"),
    )
