from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.exchanges.coinbase.constants import BTCUSD
from quant_tick.exchanges.coinbase.controllers import CoinbaseTrades


class CoinbaseTradesTest(SimpleTestCase):
    def get_trade(self, trade_id: int, timestamp: datetime) -> dict:
        return {
            "uid": str(trade_id),
            "timestamp": timestamp,
            "nanoseconds": 0,
            "price": Decimal("100"),
            "volume": Decimal("100"),
            "notional": Decimal("1"),
            "tickRule": 1,
            "index": trade_id,
        }

    def test_sequence_gap_reports_sentry_error(self):
        ts_from = datetime(2026, 4, 29, tzinfo=UTC)
        trades = [
            self.get_trade(1, ts_from),
            self.get_trade(3, ts_from + pd.Timedelta("1min")),
        ]
        controller = CoinbaseTrades.__new__(CoinbaseTrades)
        controller.symbol = SimpleNamespace(api_symbol=BTCUSD)

        with (
            patch("quant_tick.exchanges.coinbase.controllers.sentry_sdk") as sentry,
            self.assertLogs(
                "quant_tick.exchanges.coinbase.controllers", level="WARNING"
            ) as logs,
        ):
            controller.assert_data_frame(
                ts_from,
                ts_from + pd.Timedelta("2min"),
                pd.DataFrame(trades),
                trades,
            )

        self.assertIn("Coinbase trade sequence gap", logs.output[0])
        sentry.capture_message.assert_called_once()
        message = sentry.capture_message.call_args.args[0]
        self.assertIn(message, logs.output[0])
        self.assertEqual(sentry.capture_message.call_args.kwargs, {"level": "error"})
