from rest_framework.generics import ListAPIView

from quant_candles.models import Symbol
from quant_candles.serializers import SymbolSerializer


class SymbolView(ListAPIView):
    queryset = Symbol.objects.all()
    serializer_class = SymbolSerializer
