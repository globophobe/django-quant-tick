import logging
from typing import List, Union
from unittest.mock import MagicMock

from django.contrib.auth import get_user_model

from quant_tick.constants import Exchange
from quant_tick.models import GlobalSymbol, Symbol


class BaseViewTest:
    def setUp(self):
        User = get_user_model()
        user = User.objects.create(username="test")
        self.client.force_authenticate(user)


class BaseTradeViewTest(BaseViewTest):
    def setUp(self):
        super().setUp()
        self.url = self.get_url()

        logger = logging.getLogger("django.request")
        self.previous_level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR)

    def tearDown(self):
        logger = logging.getLogger("django.request")
        logger.setLevel(self.previous_level)

    def get_url(self, exchange: str) -> str:
        """Get URL."""
        raise NotImplementedError

    def get_symbols(
        self, symbols: Union[List[str], str], exchange: Exchange = Exchange.COINBASE
    ) -> dict:
        """Get symbols."""
        if isinstance(symbols, str):
            symbols = [symbols]

        global_symbol = GlobalSymbol.objects.create(name="global-symbol")
        for symbol in symbols:
            Symbol.objects.create(
                global_symbol=global_symbol,
                exchange=exchange,
                api_symbol=symbol,
            )

        return {"symbol": symbols}

    def get_mock_params(self, mock_api: MagicMock) -> List[dict]:
        """Get mock params."""
        return [mock_call.args for mock_call in mock_api.mock_calls]

    def get_mock_symbols(self, mock_api: MagicMock) -> List[str]:
        """Get mock symbols."""
        mock_params = self.get_mock_params(mock_api)
        return [mock_param[0].api_symbol for mock_param in mock_params]
