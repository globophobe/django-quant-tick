from rest_framework.request import Request
from rest_framework.response import Response

from quant_werks.exchanges import api

from .base import BaseSymbolView


class TradeView(BaseSymbolView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get data for each symbol."""
        for k in self.get_command_kwargs(request):
            api(**k)
        return Response({"ok": True})
