from django.db import models

from cryptofeed_werks.utils import gettext_lazy as _

from .base import NameMixin


class APIClient(NameMixin):
    symbols = models.ManyToManyField(
        "cryptofeed_werks.Symbol",
        db_table="cryptofeed_werks_api_client_symbol",
        verbose_name=_("symbols"),
    )
    url = models.URLField(_("url"), blank=True)
    active = models.BooleanField(_("active"), default=True)

    class Meta:
        db_table = "cryptofeed_werks_api_client"
        ordering = ("name",)
        verbose_name = _("api client")
        verbose_name_plural = _("api clients")
