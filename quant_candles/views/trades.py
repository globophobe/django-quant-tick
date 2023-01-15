from rest_framework.request import Request
from rest_framework.response import Response

from quant_candles.controllers import aggregate_trade_summary
from quant_candles.exchanges import api

from .base import BaseSymbolView


class TradeSummaryView(BaseSymbolView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        """Aggregate trade summary for each symbol."""
        for k in self.get_command_kwargs(request):
            aggregate_trade_summary(**k)
        return Response({"ok": True})


class TradeView(BaseSymbolView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get trades for each symbol."""
        for k in self.get_command_kwargs(request):
            api(**k)
        return Response({"ok": True})
