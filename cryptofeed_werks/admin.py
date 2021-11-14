from django.contrib import admin
from django.contrib.auth.models import Group, User
from django.db import models
from rest_framework.authtoken.models import TokenProxy
from semantic_admin import SemanticModelAdmin

from .models import GlobalSymbol, Symbol
from .utils import gettext_lazy as _

admin.site.unregister(User)
admin.site.unregister(Group)
admin.site.unregister(TokenProxy)


@admin.register(GlobalSymbol)
class GlobalSymbol(SemanticModelAdmin):
    pass


@admin.register(Symbol)
class Symbol(SemanticModelAdmin):
    list_display = (
        "__str__",
        "global_symbol",
        "symbol_type",
        "min_volume",
        "ok",
    )
    fields = (
        "global_symbol",
        ("exchange", "name"),
        ("symbol_type", "min_volume"),
        "ok",
    )
    readonly_fields = ("ok",)

    def ok(self, obj: models.Model) -> bool:
        return None

    ok.short_description = _("ok")
    ok.boolean = True

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related("global_symbol")
            .prefetch_related("futures")
        )
