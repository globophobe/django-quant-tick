from django.contrib import admin
from django.contrib.auth.models import Group, User
from rest_framework.authtoken.models import TokenProxy
from semantic_admin import SemanticModelAdmin, SemanticTabularInline

from .models import GlobalSymbol, Symbol

admin.site.unregister(User)
admin.site.unregister(Group)
admin.site.unregister(TokenProxy)


class SymbolInline(SemanticTabularInline):
    model = Symbol
    extra = 0


@admin.register(GlobalSymbol)
class GlobalSymbolAdmin(SemanticModelAdmin):
    list_display = ("__str__",)
    fields = ("name",)
    inlines = (SymbolInline,)


@admin.register(Symbol)
class SymbolAdmin(SemanticModelAdmin):
    list_display = ("__str__",)
    fields = (
        ("global_symbol", "exchange"),
        ("symbol_type", "api_symbol"),
        ("should_aggregate_trades", "significant_trade_filter"),
    )
