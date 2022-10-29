from rest_framework.request import Request
from rest_framework.response import Response

from quant_candles.storage import convert_trade_data_to_hourly

from .base import BaseSymbolView


class ConvertTradeDataToHourlyView(BaseSymbolView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        """Convert data from minute to hourly."""
        for k in self.get_command_kwargs(request):
            convert_trade_data_to_hourly(**k)
        return Response({"ok": True})
