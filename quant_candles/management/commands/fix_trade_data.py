import logging

from quant_candles.management.base import BaseTradeDataCommand
from quant_candles.models import TradeData
from quant_candles.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


class Command(BaseTradeDataCommand):
    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        if kwargs:
            queryset = TradeData.objects.filter(symbol=kwargs["symbol"])
            count = 0
            fixed = 0
            total = queryset.count()
            for obj in queryset:
                if obj.json_data is not None:
                    values = obj.json_data.values()
                    some_false = True in [isinstance(v, dict) for v in values]
                    some_none = None in values
                    if some_false or some_none:
                        if some_false:
                            obj.ok = False
                        else:
                            obj.ok = None
                        obj.save()
                        fixed += 1
                elif obj.ok is False:
                    obj.ok = None
                    obj.save()
                    fixed += 1

                count += 1
                logging.info(
                    _("Checked {count}/{total} objects").format(
                        **{"count": count, "total": total}
                    )
                )

            logging.info(_("Fixed {fixed} objects").format(**{"fixed": fixed}))
