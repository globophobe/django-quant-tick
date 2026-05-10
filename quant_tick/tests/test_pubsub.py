from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from quant_tick.constants import Exchange
from quant_tick.models import TradeData
from quant_tick.pubsub import (
    AGGREGATED_TRADES,
    RAW_TRADES,
    SIGNIFICANT_TRADE_FILTER_ATTR,
    SIGNIFICANT_TRADES,
    get_trade_pubsub_configs,
    get_trade_subscription_filter,
    get_trade_subscription_id,
    ingest_trades_from_pubsub,
)

from .base import BaseWriteTradeDataTest


@dataclass
class FakePubSubMessage:
    data: bytes
    attributes: dict[str, str]


@dataclass
class FakeReceivedMessage:
    ack_id: str
    message: FakePubSubMessage


@dataclass
class FakePullResponse:
    received_messages: list[FakeReceivedMessage]


class FakeSubscriber:
    def __init__(
        self,
        messages: list[FakeReceivedMessage] | dict[str, list[FakeReceivedMessage]],
    ) -> None:
        self.messages = messages
        self.ack_ids = []
        self.released_ack_ids = []

    def pull(self, *, subscription, max_messages, **_kwargs):
        if isinstance(self.messages, dict):
            messages = self.messages.setdefault(subscription, [])
        else:
            messages = self.messages
        batch = messages[:max_messages]
        del messages[:max_messages]
        return FakePullResponse(batch)

    def acknowledge(self, *, ack_ids, **_kwargs):
        self.ack_ids.extend(ack_ids)

    def modify_ack_deadline(self, *, ack_ids, ack_deadline_seconds, **_kwargs):
        if ack_deadline_seconds == 0:
            self.released_ack_ids.extend(ack_ids)


class PubSubIngestionTests(BaseWriteTradeDataTest, TestCase):
    def test_trade_subscription_contract_uses_stream_symbol_filter(self):
        symbol = self.get_symbol(
            exchange=Exchange.BITFINEX,
            api_symbol="tBTCF0:USTF0",
            save_raw=True,
            save_aggregated=True,
            significant_trade_filter=1000,
        )

        self.assertEqual(
            get_trade_subscription_id(RAW_TRADES, symbol),
            "bitfinex-tbtcf0-ustf0-raw-trades",
        )
        self.assertEqual(
            get_trade_subscription_filter(RAW_TRADES, symbol),
            'attributes.exchange="bitfinex" '
            'AND attributes.symbol="tBTCF0:USTF0"',
        )
        self.assertEqual(
            get_trade_subscription_id(AGGREGATED_TRADES, symbol),
            "bitfinex-tbtcf0-ustf0-aggregated-trades",
        )
        self.assertEqual(
            get_trade_subscription_id(SIGNIFICANT_TRADES, symbol),
            "bitfinex-tbtcf0-ustf0-significant-trades-1000",
        )
        self.assertEqual(
            get_trade_subscription_filter(SIGNIFICANT_TRADES, symbol),
            'attributes.exchange="bitfinex" '
            'AND attributes.symbol="tBTCF0:USTF0" '
            'AND attributes.significant_trade_filter="1000"',
        )

    def test_trade_pubsub_configs_return_supported_symbol_subscriptions(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            save_raw=True,
            save_aggregated=True,
            significant_trade_filter=1000,
        )

        with patch("quant_tick.pubsub.get_default_project_id", return_value="test"):
            self.assertEqual(
                get_trade_pubsub_configs(symbol),
                [
                    (
                        RAW_TRADES,
                        "projects/test/subscriptions/coinbase-btc-usd-raw-trades",
                    ),
                    (
                        AGGREGATED_TRADES,
                        "projects/test/subscriptions/"
                        "coinbase-btc-usd-aggregated-trades",
                    ),
                    (
                        SIGNIFICANT_TRADES,
                        "projects/test/subscriptions/"
                        "coinbase-btc-usd-significant-trades-1000",
                    ),
                ],
            )
            self.assertEqual(
                get_trade_pubsub_configs(symbol, AGGREGATED_TRADES),
                [
                    (
                        AGGREGATED_TRADES,
                        "projects/test/subscriptions/"
                        "coinbase-btc-usd-aggregated-trades",
                    )
                ],
            )

    def significant_trade_message(
        self,
        significant_trade_filter: str | None = "1000",
        **overrides,
    ) -> FakePubSubMessage:
        payload = {
            "exchange": Exchange.COINBASE,
            "uid": "pubsub-1",
            "symbol": "BTC-USD",
            "timestamp": (
                self.timestamp_from + pd.Timedelta("10s")
            ).isoformat().replace("+00:00", "Z"),
            "nanoseconds": 0,
            "price": "100",
            "volume": "1000",
            "notional": "10",
            "tickRule": 1,
            "ticks": 1,
            "high": "101",
            "low": "99",
            "totalBuyVolume": "1000",
            "totalVolume": "1000",
            "totalBuyNotional": "10",
            "totalNotional": "10",
            "totalBuyTicks": 1,
            "totalTicks": 1,
        }
        payload.update(overrides)
        attributes = {
            "exchange": payload["exchange"],
            "symbol": payload["symbol"],
        }
        if significant_trade_filter is not None:
            attributes[SIGNIFICANT_TRADE_FILTER_ATTR] = significant_trade_filter
        return FakePubSubMessage(
            data=json.dumps(payload).encode(),
            attributes=attributes,
        )

    def raw_trade_message(self, **overrides) -> FakePubSubMessage:
        message = self.significant_trade_message(**overrides)
        payload = json.loads(message.data)
        for key in (
            "high",
            "low",
            "totalBuyVolume",
            "totalVolume",
            "totalBuyNotional",
            "totalNotional",
            "totalBuyTicks",
            "totalTicks",
        ):
            payload.pop(key)
        attributes = dict(message.attributes)
        attributes.pop(SIGNIFICANT_TRADE_FILTER_ATTR, None)
        return FakePubSubMessage(
            data=json.dumps(payload).encode(),
            attributes=attributes,
        )

    def test_ingest_significant_trades_writes_matching_messages(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
        )
        subscriber = FakeSubscriber(
            [
                FakeReceivedMessage("ack-1", self.significant_trade_message()),
                FakeReceivedMessage(
                    "ack-2",
                    self.significant_trade_message(
                        exchange=Exchange.BINANCE,
                        symbol="BTCUSDT",
                    ),
                ),
            ]
        )

        def candle_fetcher(_symbol, timestamp_from, _timestamp_to, **_kwargs):
            return pd.DataFrame(
                [{"timestamp": timestamp_from, "notional": Decimal("10")}]
            ).set_index("timestamp")

        result = ingest_trades_from_pubsub(
            symbol=symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("1min"),
            configs=[
                (SIGNIFICANT_TRADES, "projects/test/subscriptions/significant-trades")
            ],
            subscriber=subscriber,
            candle_fetcher=candle_fetcher,
        )

        self.assertEqual(result.pulled, 2)
        self.assertEqual(result.processed, 1)
        self.assertEqual(result.ok, 1)
        self.assertEqual(subscriber.ack_ids, ["ack-1"])
        self.assertEqual(subscriber.released_ack_ids, ["ack-2"])
        trade_data = TradeData.objects.get()
        self.assertTrue(trade_data.ok)
        self.assertTrue(trade_data.filtered_data)

    def test_ingest_significant_trades_ignores_filter_mismatch(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
        )
        subscriber = FakeSubscriber(
            [
                FakeReceivedMessage(
                    "ack-1",
                    self.significant_trade_message(significant_trade_filter="2000"),
                )
            ]
        )

        def candle_fetcher(*_args, **_kwargs):
            raise AssertionError("mismatched significant-trade filter should not write")

        result = ingest_trades_from_pubsub(
            symbol=symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("1min"),
            configs=[
                (SIGNIFICANT_TRADES, "projects/test/subscriptions/significant-trades")
            ],
            subscriber=subscriber,
            candle_fetcher=candle_fetcher,
        )

        self.assertEqual(result.pulled, 1)
        self.assertEqual(result.processed, 0)
        self.assertEqual(result.ignored, 1)
        self.assertEqual(subscriber.ack_ids, ["ack-1"])
        self.assertFalse(TradeData.objects.exists())

    def test_ingest_raw_trades_uses_standard_trade_data_write(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            save_raw=True,
            save_aggregated=True,
            significant_trade_filter=1000,
        )
        subscriber = FakeSubscriber(
            [FakeReceivedMessage("ack-1", self.raw_trade_message())]
        )

        def candle_fetcher(_symbol, timestamp_from, _timestamp_to, **_kwargs):
            return pd.DataFrame(
                [{"timestamp": timestamp_from, "notional": Decimal("10")}]
            ).set_index("timestamp")

        result = ingest_trades_from_pubsub(
            symbol=symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("1min"),
            configs=[(RAW_TRADES, "projects/test/subscriptions/raw-trades")],
            subscriber=subscriber,
            candle_fetcher=candle_fetcher,
        )

        self.assertEqual(result.ok, 1)
        trade_data = TradeData.objects.get()
        self.assertTrue(trade_data.ok)
        self.assertTrue(trade_data.raw_data)
        self.assertTrue(trade_data.aggregated_data)
        self.assertTrue(trade_data.filtered_data)

    def test_ingest_aggregated_trades_writes_prepared_aggregated_data(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            save_raw=True,
            save_aggregated=True,
            significant_trade_filter=1000,
        )
        subscriber = FakeSubscriber(
            [FakeReceivedMessage("ack-1", self.raw_trade_message())]
        )

        def candle_fetcher(_symbol, timestamp_from, _timestamp_to, **_kwargs):
            return pd.DataFrame(
                [{"timestamp": timestamp_from, "notional": Decimal("10")}]
            ).set_index("timestamp")

        result = ingest_trades_from_pubsub(
            symbol=symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("1min"),
            configs=[
                (AGGREGATED_TRADES, "projects/test/subscriptions/aggregated-trades")
            ],
            subscriber=subscriber,
            candle_fetcher=candle_fetcher,
        )

        self.assertEqual(result.ok, 1)
        trade_data = TradeData.objects.get()
        self.assertTrue(trade_data.ok)
        self.assertFalse(trade_data.raw_data)
        self.assertTrue(trade_data.aggregated_data)
        self.assertTrue(trade_data.filtered_data)

    def test_ingest_writes_all_available_streams_together(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            save_raw=True,
            save_aggregated=True,
            significant_trade_filter=1000,
        )
        subscriber = FakeSubscriber(
            {
                "raw-sub": [FakeReceivedMessage("raw-1", self.raw_trade_message())],
                "aggregated-sub": [
                    FakeReceivedMessage(
                        "aggregated-1",
                        self.significant_trade_message(significant_trade_filter=None),
                    )
                ],
                "significant-sub": [
                    FakeReceivedMessage("significant-1", self.significant_trade_message())
                ],
            }
        )

        def candle_fetcher(_symbol, timestamp_from, _timestamp_to, **_kwargs):
            return pd.DataFrame(
                [{"timestamp": timestamp_from, "notional": Decimal("10")}]
            ).set_index("timestamp")

        result = ingest_trades_from_pubsub(
            symbol=symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("1min"),
            configs=[
                (RAW_TRADES, "raw-sub"),
                (AGGREGATED_TRADES, "aggregated-sub"),
                (SIGNIFICANT_TRADES, "significant-sub"),
            ],
            subscriber=subscriber,
            candle_fetcher=candle_fetcher,
        )

        self.assertEqual(result.pulled, 3)
        self.assertEqual(result.processed, 3)
        self.assertEqual(result.ok, 3)
        self.assertEqual(
            sorted(subscriber.ack_ids),
            ["aggregated-1", "raw-1", "significant-1"],
        )
        trade_data = TradeData.objects.get()
        self.assertTrue(trade_data.ok)
        self.assertTrue(trade_data.raw_data)
        self.assertTrue(trade_data.aggregated_data)
        self.assertTrue(trade_data.filtered_data)

    def test_ingest_drains_subscription_in_batches(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            save_raw=True,
        )
        subscriber = FakeSubscriber(
            [
                FakeReceivedMessage("ack-1", self.raw_trade_message()),
                FakeReceivedMessage("ack-2", self.raw_trade_message(uid="pubsub-2")),
            ]
        )

        def candle_fetcher(_symbol, timestamp_from, _timestamp_to, **_kwargs):
            return pd.DataFrame(
                [{"timestamp": timestamp_from, "notional": Decimal("20")}]
            ).set_index("timestamp")

        result = ingest_trades_from_pubsub(
            symbol=symbol,
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("1min"),
            configs=[(RAW_TRADES, "raw-sub")],
            batch_size=1,
            subscriber=subscriber,
            candle_fetcher=candle_fetcher,
        )

        self.assertEqual(result.pulled, 2)
        self.assertEqual(result.processed, 2)
        self.assertEqual(result.ok, 2)
        self.assertEqual(subscriber.ack_ids, ["ack-1", "ack-2"])
