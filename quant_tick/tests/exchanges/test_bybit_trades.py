from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.bybit.controllers import (
    BybitSpotTradesS3,
    BybitTradesS3,
    BybitTradesWebSocket,
    bybit_trades,
)


class BybitTradesTest(SimpleTestCase):
    def get_controller(
        self,
        api_symbol="BTCUSDT",
        exchange=Exchange.BYBIT_LINEAR,
    ):
        controller = BybitTradesS3.__new__(BybitTradesS3)
        controller.symbol = SimpleNamespace(
            exchange=exchange,
            api_symbol=api_symbol,
            symbol_type=SymbolType.PERPETUAL,
        )
        return controller

    def test_s3_uses_exact_daily_archive_url(self):
        url = self.get_controller().get_url(date(2026, 7, 23))

        self.assertEqual(
            url,
            "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2026-07-23.csv.gz",
        )

    def test_spot_s3_uses_exact_daily_archive_url(self):
        controller = BybitSpotTradesS3.__new__(BybitSpotTradesS3)
        controller.symbol = SimpleNamespace(api_symbol="BTCUSDT")

        self.assertEqual(
            controller.get_url(date(2026, 7, 23)),
            "https://public.bybit.com/spot/BTCUSDT/BTCUSDT_2026-07-23.csv.gz",
        )

    def test_trades_use_archive_then_websocket_without_publication_clamp(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 7, 23, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(days=2)
        on_data_frame = Mock()

        calls = []
        with (
            patch("quant_tick.exchanges.bybit.controllers.BybitTradesS3") as archive,
            patch(
                "quant_tick.exchanges.bybit.controllers.BybitTradesWebSocket"
            ) as websocket,
        ):
            archive.return_value.main.side_effect = lambda: calls.append("archive")
            websocket.return_value.main.side_effect = lambda: calls.append("websocket")
            bybit_trades(
                symbol,
                timestamp_from,
                timestamp_to,
                on_data_frame,
            )

        self.assertEqual(calls, ["archive", "websocket"])
        for controller in (archive, websocket):
            controller.assert_called_once_with(
                symbol,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                on_data_frame=on_data_frame,
                retry=False,
                verbose=False,
            )
            controller.return_value.main.assert_called_once_with()

    def test_websocket_controller_promotes_valid_current_minute(self):
        timestamp_from = datetime(2026, 7, 23, 12, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=1)
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
            save_raw=False,
            save_aggregated=False,
            significant_trade_filter=1000,
        )
        on_data_frame = Mock()
        filtered = pd.DataFrame([{"uid": "trade"}])
        candles = pd.DataFrame([])
        controller = BybitTradesWebSocket(
            symbol,
            timestamp_from,
            timestamp_to,
            on_data_frame,
        )
        controller.get_websocket_timestamp_from = Mock(return_value=timestamp_from)
        controller.get_candles = Mock(return_value=candles)
        controller.validate_websocket_partitions = Mock(
            return_value={timestamp_from: (None, None, filtered)}
        )

        with patch(
            "quant_tick.controllers.rest.TradeData.objects.overlapping"
        ) as overlapping:
            overlapping.return_value.exists.return_value = False
            controller.main()

        on_data_frame.assert_called_once_with(
            symbol,
            timestamp_from,
            timestamp_to,
            filtered,
            candles,
            filtered_trades=filtered,
        )

    def test_websocket_controller_promotes_newest_minute_first(self):
        timestamp_from = datetime(2026, 7, 23, 12, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=2)
        newest = timestamp_from + timedelta(minutes=1)
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
            save_raw=False,
            save_aggregated=False,
            significant_trade_filter=1000,
        )
        on_data_frame = Mock()
        old_filtered = pd.DataFrame([{"uid": "old"}])
        new_filtered = pd.DataFrame([{"uid": "new"}])
        controller = BybitTradesWebSocket(
            symbol,
            timestamp_from,
            timestamp_to,
            on_data_frame,
        )
        controller.get_websocket_timestamp_from = Mock(return_value=timestamp_from)
        controller.get_candles = Mock(return_value=pd.DataFrame([]))
        controller.validate_websocket_partitions = Mock(
            return_value={
                timestamp_from: (None, None, old_filtered),
                newest: (None, None, new_filtered),
            }
        )

        with patch(
            "quant_tick.controllers.rest.TradeData.objects.overlapping"
        ) as overlapping:
            overlapping.return_value.exists.return_value = False
            controller.main()

        self.assertEqual(
            [(call.args[1], call.args[2]) for call in on_data_frame.call_args_list],
            [
                (newest, timestamp_to),
                (timestamp_from, newest),
            ],
        )

    def test_websocket_controller_preserves_larger_archive_partition(self):
        timestamp_from = datetime(2026, 7, 23, 12, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=1)
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
            save_raw=False,
            save_aggregated=False,
            significant_trade_filter=1000,
        )
        on_data_frame = Mock()
        filtered = pd.DataFrame([{"uid": "trade"}])
        controller = BybitTradesWebSocket(
            symbol,
            timestamp_from,
            timestamp_to,
            on_data_frame,
        )
        controller.get_websocket_timestamp_from = Mock(return_value=timestamp_from)
        controller.get_candles = Mock(return_value=pd.DataFrame([]))
        controller.validate_websocket_partitions = Mock(
            return_value={timestamp_from: (None, None, filtered)}
        )

        with patch(
            "quant_tick.controllers.rest.TradeData.objects.overlapping"
        ) as overlapping:
            overlapping.return_value.exists.return_value = True
            controller.main()

        on_data_frame.assert_not_called()

    def test_spot_trades_use_spot_archive_then_websocket(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT,
        )
        calls = []
        with (
            patch(
                "quant_tick.exchanges.bybit.controllers.BybitSpotTradesS3"
            ) as archive,
            patch(
                "quant_tick.exchanges.bybit.controllers.BybitTradesWebSocket"
            ) as websocket,
        ):
            archive.return_value.main.side_effect = lambda: calls.append("archive")
            websocket.return_value.main.side_effect = lambda: calls.append("websocket")
            bybit_trades(
                symbol,
                datetime(2026, 7, 23, tzinfo=UTC),
                datetime(2026, 7, 24, tzinfo=UTC),
                Mock(),
            )

        self.assertEqual(calls, ["archive", "websocket"])
        archive.assert_called_once()
        archive.return_value.main.assert_called_once_with()
        websocket.assert_called_once()
        websocket.return_value.main.assert_called_once_with()

    def test_spot_archive_normalizes_chunked_hours(self):
        first_hour = datetime(2022, 11, 10, tzinfo=UTC)
        second_hour = first_hour + timedelta(hours=1)
        timestamp_to = second_hour + timedelta(hours=1)
        data = pd.DataFrame(
            [
                {
                    "id": "1",
                    "timestamp": "1668038400525",
                    "price": "15910.61",
                    "volume": "0.003383",
                    "side": "sell",
                },
                {
                    "id": "2",
                    "timestamp": "1668038400526",
                    "price": "15910.30",
                    "volume": "0.0001",
                    "side": "buy",
                },
                {
                    "id": "3",
                    "timestamp": "1668042000000",
                    "price": "16000",
                    "volume": "0.001",
                    "side": "buy",
                },
            ]
        )
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT,
        )
        on_data_frame = Mock()
        controller = BybitSpotTradesS3(
            symbol,
            timestamp_from=first_hour,
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
        )
        controller.get_data_frame_chunks = Mock(
            return_value=iter([data.iloc[:1].copy(), data.iloc[1:].copy()])
        )
        controller.get_candles = Mock(return_value=pd.DataFrame([]))

        with (
            patch(
                "quant_tick.controllers.s3.TradeDataIterator.iter_days",
                return_value=[(first_hour, timestamp_to, [])],
            ),
            patch(
                "quant_tick.controllers.s3.TradeDataIterator.iter_hours",
                return_value=[
                    (second_hour, timestamp_to),
                    (first_hour, second_hour),
                ],
            ),
        ):
            controller.main()

        self.assertEqual(
            [(call.args[1], call.args[2]) for call in on_data_frame.call_args_list],
            [
                (first_hour, second_hour),
                (second_hour, timestamp_to),
            ],
        )
        first = on_data_frame.call_args_list[0].args[3]
        second = on_data_frame.call_args_list[1].args[3]
        self.assertEqual(first["uid"].tolist(), ["1", "2"])
        self.assertEqual(first["tickRule"].tolist(), [-1, 1])
        self.assertEqual(
            first["notional"].tolist(),
            [Decimal("0.003383"), Decimal("0.0001")],
        )
        self.assertEqual(
            first["volume"].tolist(),
            [Decimal("53.82559363"), Decimal("1.591030")],
        )
        self.assertEqual(first["nanoseconds"].tolist(), [0, 0])
        self.assertEqual(second["uid"].tolist(), ["3"])

    def test_linear_archive_normalizes_chunked_hours(self):
        first_hour = datetime(2020, 1, 1, tzinfo=UTC)
        second_hour = first_hour + timedelta(hours=1)
        timestamp_to = second_hour + timedelta(hours=1)
        data = pd.DataFrame(
            [
                {
                    "timestamp": "1577836800.0200",
                    "side": "Sell",
                    "size": "0.072",
                    "price": "6698",
                    "trdMatchID": "earlier",
                    "symbol": "BTCUSDT",
                },
                {
                    "timestamp": "1577836800.0647",
                    "side": "Buy",
                    "size": "0.042",
                    "price": "6698.5",
                    "trdMatchID": "later",
                    "symbol": "BTCUSDT",
                },
                {
                    "timestamp": "1577840400.0100",
                    "side": "Buy",
                    "size": "0.001",
                    "price": "6700",
                    "trdMatchID": "second-hour",
                    "symbol": "BTCUSDT",
                },
            ]
        )
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        on_data_frame = Mock()
        controller = BybitTradesS3(
            symbol,
            timestamp_from=first_hour,
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
        )
        controller.get_data_frame_chunks = Mock(
            return_value=iter([data.iloc[:1].copy(), data.iloc[1:].copy()])
        )
        controller.get_candles = Mock(return_value=pd.DataFrame([]))

        with (
            patch(
                "quant_tick.controllers.s3.TradeDataIterator.iter_days",
                return_value=[(first_hour, timestamp_to, [])],
            ),
            patch(
                "quant_tick.controllers.s3.TradeDataIterator.iter_hours",
                return_value=[
                    (second_hour, timestamp_to),
                    (first_hour, second_hour),
                ],
            ),
        ):
            controller.main()

        self.assertEqual(
            [(call.args[1], call.args[2]) for call in on_data_frame.call_args_list],
            [
                (first_hour, second_hour),
                (second_hour, timestamp_to),
            ],
        )
        first = on_data_frame.call_args_list[0].args[3]
        second = on_data_frame.call_args_list[1].args[3]
        self.assertEqual(first["uid"].tolist(), ["earlier", "later"])
        self.assertEqual(first["tickRule"].tolist(), [-1, 1])
        self.assertEqual(
            first["notional"].tolist(),
            [Decimal("0.072"), Decimal("0.042")],
        )
        self.assertEqual(
            first["volume"].tolist(),
            [Decimal("482.256"), Decimal("281.3370")],
        )
        self.assertEqual(first.iloc[1]["timestamp"].microsecond, 64700)
        self.assertEqual(first.iloc[1]["nanoseconds"], 0)
        self.assertEqual(second["uid"].tolist(), ["second-hour"])
        with self.assertRaisesRegex(ValueError, "rows outside BTCUSDT"):
            controller.prepare_archive_chunk(
                pd.DataFrame([{"symbol": "ETHUSDT"}])
            )

    def test_archive_preserves_source_order_for_exact_timestamp_ties(self):
        rows = []
        for uid in (
            "bbb107e9-f7e8-53f3-b18a-ac9781c6eae8",
            "aaa107e9-f7e8-53f3-b18a-ac9781c6eae8",
        ):
            rows.append(
                {
                    "timestamp": "1585180700.0647",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "size": "0.001",
                    "price": "6698.5",
                    "tickDirection": "PlusTick",
                    "trdMatchID": uid,
                    "grossValue": "669850000",
                    "foreignNotional": "6.6985",
                }
            )

        parsed = self.get_controller().parse_dtypes_and_strip_columns(
            pd.DataFrame(rows)
        )

        self.assertEqual(
            parsed["uid"].tolist(),
            [
                "bbb107e9-f7e8-53f3-b18a-ac9781c6eae8",
                "aaa107e9-f7e8-53f3-b18a-ac9781c6eae8",
            ],
        )

    def test_inverse_archive_normalizes_contracts_to_base_notional(self):
        data = pd.DataFrame(
            [
                {
                    "timestamp": "1585180700",
                    "symbol": "BTCUSD",
                    "side": "Buy",
                    "size": "100",
                    "price": "10000",
                    "tickDirection": "PlusTick",
                    "trdMatchID": "trade",
                    "grossValue": "1000000",
                    "foreignNotional": "100",
                }
            ]
        )

        parsed = self.get_controller(
            "BTCUSD",
            Exchange.BYBIT_INVERSE,
        ).parse_dtypes_and_strip_columns(data)

        self.assertEqual(parsed.iloc[0]["volume"], Decimal("100"))
        self.assertEqual(parsed.iloc[0]["notional"], Decimal("0.01"))
