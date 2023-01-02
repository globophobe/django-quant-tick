import string

import pandas as pd

from quant_candles.constants import Exchange
from quant_candles.models import Candle, Symbol, TradeData

from .base import BaseWriteTradeDataTest


class CandleTest(BaseWriteTradeDataTest):
    def setUp(self):
        super().setUp()
        self.candle = Candle.objects.create()

    def get_symbol(self, name: str, exchange: Exchange = Exchange.COINBASE) -> Symbol:
        """Get symbol."""
        return Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=exchange,
            api_symbol=name,
        )

    def test_get_data_frame_with_one_symbol(self):
        """Get data frame with one symbol."""
        symbol = self.get_symbol("test")
        self.candle.symbols.add(symbol)
        filtered = self.get_filtered(self.timestamp_from)
        TradeData.write(symbol, self.timestamp_from, self.timestamp_to, filtered, {})
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        data_frame = t.get_data_frame()
        df = self.candle.get_data_frame(self.timestamp_from, self.timestamp_to)
        for column in data_frame.columns:
            with self.subTest(column=column):
                self.assertEqual(data_frame.iloc[0][column], df.iloc[0][column])

    def test_get_data_frame_with_two_symbols(self):
        """Get data frame with two symbols."""
        symbols = [
            self.get_symbol(f"test-{letter}") for letter in string.ascii_uppercase[:2]
        ]
        self.candle.symbols.add(*symbols)
        for index, symbol in enumerate(symbols):
            filtered = self.get_filtered(self.timestamp_from)
            TradeData.write(
                symbol, self.timestamp_from, self.timestamp_to, filtered, {}
            )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 2)
        data_frame = pd.concat([t.get_data_frame() for t in trade_data])
        df = self.candle.get_data_frame(self.timestamp_from, self.timestamp_to)
        self.assertEqual(len(data_frame), len(df))
        for row, r in zip(data_frame.itertuples(), df.itertuples()):
            for column in data_frame.columns:
                with self.subTest(column=column, row=row):
                    self.assertEqual(getattr(row, column), getattr(r, column))

    def test_get_sorted_data_frame_with_two_symbols(self):
        """Get sorted data frame with two symbols."""
        symbols = [
            self.get_symbol(f"test-{letter}") for letter in string.ascii_uppercase[:2]
        ]
        self.candle.symbols.add(*symbols)
        for index, symbol in enumerate(symbols):
            nanoseconds = 1 if index == 0 else 0
            filtered = self.get_filtered(self.timestamp_from, nanoseconds=nanoseconds)
            TradeData.write(
                symbol, self.timestamp_from, self.timestamp_to, filtered, {}
            )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 2)
        data_frame = pd.concat([t.get_data_frame() for t in trade_data])
        df = self.candle.get_data_frame(self.timestamp_from, self.timestamp_to)
        self.assertEqual(len(data_frame), len(df))
        for row, r in zip(data_frame.iloc[::-1].itertuples(), df.itertuples()):
            for column in data_frame.columns:
                with self.subTest(column=column, row=row):
                    self.assertEqual(getattr(row, column), getattr(r, column))
