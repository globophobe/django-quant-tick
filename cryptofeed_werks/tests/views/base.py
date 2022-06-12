from typing import List, Union
from unittest.mock import MagicMock

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.models import GlobalSymbol, Symbol


class BaseViewTest(APITestCase):
    def setUp(self):
        User = get_user_model()
        user = User.objects.create(username="test")
        self.client.force_authenticate(user)
        self.url = self.get_url()

    def get_url(self, exchange: str) -> str:
        """Get URL."""
        raise NotImplementedError

    def get_symbols(
        self, symbols: Union[List[str], str], exchange: Exchange = Exchange.FTX
    ) -> dict:
        """Get symbols."""
        if isinstance(symbols, str):
            symbols = [symbols]

        global_symbol = GlobalSymbol.objects.create(name="global-symbol")
        for symbol in symbols:
            Symbol.objects.create(
                exchange=exchange,
                global_symbol=global_symbol,
                api_symbol=symbol,
            )

        return {"symbol": symbols}

    def get_mock_symbols(self, mock_api: MagicMock) -> List[str]:
        mock_params = self.get_mock_params(mock_api)
        return [mock_param["symbol"].api_symbol for mock_param in mock_params]

    def get_mock_params(self, mock_api: MagicMock) -> List[dict]:
        """Get mock params."""
        return [mock_call.kwargs for mock_call in mock_api.mock_calls]
