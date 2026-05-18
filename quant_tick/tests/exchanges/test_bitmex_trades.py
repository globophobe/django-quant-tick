import pandas as pd
from django.test import SimpleTestCase

from quant_tick.exchanges.bitmex.controllers import BitmexTradesS3


class BitmexTradesTest(SimpleTestCase):
    def test_s3_tick_rule_maps_side_values(self):
        controller = BitmexTradesS3.__new__(BitmexTradesS3)
        data = pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-10D00:00:00.000001",
                    "symbol": "XBTUSD",
                    "side": "Sell",
                    "size": "1",
                    "price": "100",
                    "tickDirection": "PlusTick",
                    "trdMatchID": "sell-plus",
                    "grossValue": "100",
                    "foreignNotional": "100",
                },
                {
                    "timestamp": "2026-05-10D00:00:01.000002",
                    "symbol": "XBTUSD",
                    "side": "Buy",
                    "size": "1",
                    "price": "100",
                    "tickDirection": "MinusTick",
                    "trdMatchID": "buy-minus",
                    "grossValue": "100",
                    "foreignNotional": "100",
                },
                {
                    "timestamp": "2026-05-10D00:00:02.000003",
                    "symbol": "XBTUSD",
                    "side": "Sell",
                    "size": "1",
                    "price": "100",
                    "tickDirection": "ZeroPlusTick",
                    "trdMatchID": "sell-zero-plus",
                    "grossValue": "100",
                    "foreignNotional": "100",
                },
                {
                    "timestamp": "2026-05-10D00:00:03.000004",
                    "symbol": "XBTUSD",
                    "side": "Buy",
                    "size": "1",
                    "price": "100",
                    "tickDirection": "ZeroMinusTick",
                    "trdMatchID": "buy-zero-minus",
                    "grossValue": "100",
                    "foreignNotional": "100",
                },
            ]
        )

        parsed = controller.parse_dtypes_and_strip_columns(data)

        self.assertEqual(parsed["tickRule"].tolist(), [-1, 1, -1, 1])
