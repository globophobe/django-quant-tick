import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.lib import (
    get_current_time,
    get_min_time,
    get_next_time,
    get_previous_time,
)
from quant_tick.models import Candle, CandleCache, TradeData
from quant_tick.storage import (
    clean_trade_data_overlaps,
    clean_unlinked_trade_data_files,
    convert_candle_cache_to_daily,
    convert_trade_data,
    convert_trade_data_to_daily,
)

from .base import BaseSymbolTest, BaseWriteTradeDataTest


class TradeDataStorageTest(BaseWriteTradeDataTest, TestCase):
    def test_clean_trade_data_overlaps_deletes_hourly_and_minute_rows(self):
        symbol = self.get_symbol()
        day_from = get_min_time(self.timestamp_from, "1d")
        hour_from = day_from + pd.Timedelta("12h")
        minute_from = hour_from + pd.Timedelta("5min")

        TradeData.objects.create(
            symbol=symbol,
            timestamp=day_from,
            frequency=Frequency.DAY,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=hour_from,
            frequency=Frequency.HOUR,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=minute_from,
            frequency=Frequency.MINUTE,
        )

        deleted = clean_trade_data_overlaps(
            symbol,
            hour_from + pd.Timedelta("1min"),
            hour_from + pd.Timedelta("2min"),
        )

        rows = list(
            TradeData.objects.filter(symbol=symbol).order_by("timestamp", "frequency")
        )
        self.assertEqual(deleted, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, day_from)
        self.assertEqual(rows[0].frequency, Frequency.DAY)

    def test_clean_trade_data_overlaps_deletes_minute_rows_covered_by_hourly_row(self):
        symbol = self.get_symbol()
        day_from = get_min_time(self.timestamp_from, "1d")
        hour_from = day_from + pd.Timedelta("6h")
        minute_from = hour_from + pd.Timedelta("5min")

        TradeData.objects.create(
            symbol=symbol,
            timestamp=hour_from,
            frequency=Frequency.HOUR,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=minute_from,
            frequency=Frequency.MINUTE,
        )

        deleted = clean_trade_data_overlaps(
            symbol,
            minute_from,
            minute_from + pd.Timedelta("1min"),
        )

        rows = list(
            TradeData.objects.filter(symbol=symbol).order_by("timestamp", "frequency")
        )
        self.assertEqual(deleted, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, hour_from)
        self.assertEqual(rows[0].frequency, Frequency.HOUR)

    def test_convert_trade_data_to_daily_compacts_complete_hour(self):
        symbol = self.get_symbol()
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        data_frames = []
        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}min")
            ts_to = ts_from + pd.Timedelta("1min")
            df = self.get_raw(ts_from)
            TradeData.write(symbol, ts_from, ts_to, pd.DataFrame([]), raw_trades=df)
            data_frames.append(df)

        source_rows = list(TradeData.objects.all())
        self.assertEqual(len(source_rows), 60)
        first = source_rows[0]
        candles = pd.DataFrame([row.json_data["candle"] for row in source_rows])
        source_files = [
            (row.raw_data.storage, row.raw_data.name) for row in source_rows
        ]

        convert_trade_data_to_daily(
            symbol, timestamp_from, get_next_time(timestamp_from, value="1h")
        )

        trades = list(TradeData.objects.all())
        self.assertEqual(len(trades), 1)

        raw = pd.concat(data_frames).drop(columns=["uid"]).reset_index(drop=True)
        data = trades[0]
        self.assertEqual(data.frequency, Frequency.HOUR)
        self.assertEqual(data.uid, first.uid)
        self.assertEqual(
            Path(data.raw_data.name).name,
            f"{timestamp_from:%H%M}-{int(Frequency.HOUR)}m.parquet",
        )
        self.assertTrue(data.get_data_frame(FileData.RAW).equals(raw))
        for storage, name in source_files:
            self.assertTrue(storage.exists(name))
        clean_unlinked_trade_data_files(
            symbol,
            timestamp_from,
            timestamp_from + pd.Timedelta("1h"),
        )
        for storage, name in source_files:
            self.assertFalse(storage.exists(name))
        candle = data.json_data["candle"]
        self.assertEqual(candle["timestamp"], candles.iloc[0].timestamp)
        self.assertEqual(candle["open"], candles.iloc[0].open)
        self.assertEqual(candle["high"], candles.high.max())
        self.assertEqual(candle["low"], candles.low.min())
        self.assertEqual(candle["close"], candles.iloc[-1].close)
        self.assertEqual(candle["volume"], candles.volume.sum())
        self.assertEqual(candle["buyVolume"], candles.buyVolume.sum())
        self.assertEqual(candle["notional"], candles.notional.sum())
        self.assertEqual(candle["buyNotional"], candles.buyNotional.sum())
        self.assertEqual(candle["ticks"], candles.ticks.sum())
        self.assertEqual(candle["buyTicks"], candles.buyTicks.sum())

    def test_convert_trade_data_to_daily_compacts_complete_day(self):
        day_from = get_min_time(self.timestamp_from, "1d")
        cases = (
            ("all-true", [True] * 24, True),
            ("true-and-none", [True] * 23 + [None], None),
            ("all-none", [None] * 24, None),
            ("false-wins", [True] * 22 + [None, False], False),
        )

        for name, validation_states, expected in cases:
            with self.subTest(name=name):
                symbol = self.get_symbol(api_symbol=name, save_raw=False)
                for hour, ok in enumerate(validation_states):
                    TradeData.objects.create(
                        symbol=symbol,
                        timestamp=day_from + pd.Timedelta(f"{hour}h"),
                        frequency=Frequency.HOUR,
                        ok=ok,
                    )

                convert_trade_data_to_daily(
                    symbol,
                    day_from,
                    day_from + pd.Timedelta("1d"),
                )

                trades = list(TradeData.objects.filter(symbol=symbol))
                self.assertEqual(len(trades), 1)
                self.assertEqual(trades[0].timestamp, day_from)
                self.assertEqual(trades[0].frequency, Frequency.DAY)
                self.assertIs(trades[0].ok, expected)

    def test_convert_trade_data_to_daily_does_not_compact_incomplete_day(self):
        symbol = self.get_symbol(save_raw=False)
        day_from = get_min_time(self.timestamp_from, "1d")

        for hour in range(4):
            TradeData.objects.create(
                symbol=symbol,
                timestamp=day_from + pd.Timedelta(f"{hour}h") + pd.Timedelta("1min"),
                frequency=Frequency.HOUR,
                ok=True,
            )
        for hour in range(4, 24):
            TradeData.objects.create(
                symbol=symbol,
                timestamp=day_from + pd.Timedelta(f"{hour}h"),
                frequency=Frequency.HOUR,
                ok=True,
            )

        convert_trade_data_to_daily(symbol, day_from, day_from + pd.Timedelta("1d"))

        trades = TradeData.objects.filter(symbol=symbol)
        self.assertFalse(trades.filter(frequency=Frequency.DAY).exists())
        self.assertEqual(trades.filter(frequency=Frequency.HOUR).count(), 24)

    def test_convert_trade_data_rolls_back_source_rows_on_failure(self):
        symbol = self.get_symbol()
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}min")
            ts_to = ts_from + pd.Timedelta("1min")
            df = self.get_raw(ts_from)
            TradeData.write(symbol, ts_from, ts_to, pd.DataFrame([]), raw_trades=df)

        queryset = TradeData.objects.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=get_next_time(timestamp_from, value="1h"),
            frequency=Frequency.MINUTE,
        )
        source_rows = list(queryset)
        source_files = [
            (row.raw_data.storage, row.raw_data.name) for row in source_rows
        ]
        target = TradeData(
            symbol=symbol,
            timestamp=timestamp_from,
            frequency=Frequency.HOUR,
        )
        target_name = target.upload_path("raw", "data.parquet")

        original_delete = TradeData.delete
        delete_calls = 0

        def fail_second_delete(instance, *args, **kwargs):
            nonlocal delete_calls
            delete_calls += 1
            if delete_calls == 2:
                raise RuntimeError("boom")
            return original_delete(instance, *args, **kwargs)

        try:
            with (
                patch.object(TradeData, "delete", new=fail_second_delete),
                self.assertRaises(RuntimeError),
            ):
                convert_trade_data(
                    symbol,
                    queryset,
                    timestamp_from,
                    get_next_time(timestamp_from, value="1h"),
                )

            self.assertEqual(delete_calls, 2)
            self.assertEqual(
                TradeData.objects.filter(
                    symbol=symbol,
                    frequency=Frequency.MINUTE,
                ).count(),
                60,
            )
            self.assertFalse(
                TradeData.objects.filter(
                    symbol=symbol,
                    frequency=Frequency.HOUR,
                ).exists()
            )
            for storage, name in source_files:
                self.assertTrue(storage.exists(name))
            self.assertTrue(source_files[0][0].exists(target_name))
        finally:
            shutil.rmtree(Path("test-trades"), ignore_errors=True)

    def test_convert_trade_data_reserves_existing_target_before_upload(self):
        symbol = self.get_symbol()
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}min")
            TradeData.write(
                symbol,
                ts_from,
                ts_from + pd.Timedelta("1min"),
                pd.DataFrame([]),
                raw_trades=self.get_raw(ts_from),
            )

        queryset = TradeData.objects.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_from + pd.Timedelta("1h"),
            frequency=Frequency.MINUTE,
        )
        target = TradeData(
            symbol=symbol,
            timestamp=timestamp_from,
            frequency=Frequency.HOUR,
        )
        target.raw_data.save("data.parquet", ContentFile(b"existing"), save=True)
        target_name = target.raw_data.name
        storage = target.raw_data.storage

        with (
            patch.object(storage, "save", wraps=storage.save) as save,
            self.assertRaises(IntegrityError),
        ):
            convert_trade_data(
                symbol,
                queryset,
                timestamp_from,
                timestamp_from + pd.Timedelta("1h"),
            )

        save.assert_not_called()
        with storage.open(target_name, "rb") as target_file:
            self.assertEqual(target_file.read(), b"existing")

    def test_unlinked_cleanup_protects_full_day_and_each_file_field(self):
        symbol = self.get_symbol(save_raw=True, save_aggregated=True)
        day_from = get_min_time(self.timestamp_from, "1d")
        raw_storage = TradeData._meta.get_field(FileData.RAW).storage

        orphan = TradeData(
            symbol=symbol,
            timestamp=day_from,
            frequency=Frequency.DAY,
        )
        orphan_name = orphan.upload_path("raw", "data.parquet")
        raw_storage.save(orphan_name, ContentFile(b"orphan"))

        early = TradeData(
            symbol=symbol,
            timestamp=day_from + pd.Timedelta("1h"),
            frequency=Frequency.HOUR,
        )
        early.raw_data.save("data.parquet", ContentFile(b"early"), save=True)

        current = TradeData(
            symbol=symbol,
            timestamp=day_from + pd.Timedelta("12h"),
            frequency=Frequency.HOUR,
        )
        current.raw_data.save("data.parquet", ContentFile(b"current"), save=False)
        current.aggregated_data.save(
            "data.parquet",
            ContentFile(b"aggregated"),
            save=True,
        )

        late = TradeData(
            symbol=symbol,
            timestamp=day_from + pd.Timedelta("14h"),
            frequency=Frequency.HOUR,
        )
        late.raw_data.save("data.parquet", ContentFile(b"late"), save=True)

        following = TradeData(
            symbol=symbol,
            timestamp=day_from + pd.Timedelta("1d"),
            frequency=Frequency.DAY,
        )
        following.raw_data.save("data.parquet", ContentFile(b"following"), save=True)

        clean_unlinked_trade_data_files(
            symbol,
            day_from + pd.Timedelta("12h"),
            day_from + pd.Timedelta("1d"),
        )

        self.assertFalse(raw_storage.exists(orphan_name))
        for row in (early, current, late, following):
            self.assertTrue(raw_storage.exists(row.raw_data.name))
        self.assertTrue(
            current.aggregated_data.storage.exists(current.aggregated_data.name)
        )


class CandleCacheStorageTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.candle = Candle.objects.create(
            symbol=self.get_symbol(), json_data={"sample_type": SampleType.NOTIONAL}
        )

    def test_convert_candle_cache_to_daily(self):
        timestamp_to = get_min_time(get_current_time(), value="1d")
        timestamp_from = get_previous_time(timestamp_to, value="1d")
        total = 24
        target_value = 25
        for value in range(total):
            ts = timestamp_from + pd.Timedelta(f"{value}h")
            val = value + 1
            expected_next = {
                "open": 0,
                "high": val,
                "low": -val,
                "close": 1,
                "volume": val * 1000,
                "buyVolume": val * 500,
                "notional": val * 100,
                "buyNotional": val * 50,
                "ticks": val * 10,
                "buyTicks": val * 5,
            }
            CandleCache.objects.create(
                candle=self.candle,
                timestamp=ts,
                frequency=Frequency.HOUR,
                json_data={
                    "sample_value": val,
                    "target_value": target_value,
                    "next": expected_next,
                },
            )
        convert_candle_cache_to_daily(self.candle)
        candle_cache = CandleCache.objects.filter(candle=self.candle)
        self.assertFalse(candle_cache.filter(frequency=Frequency.HOUR).exists())
        daily = candle_cache.filter(frequency=Frequency.DAY)
        self.assertEqual(daily.count(), 1)
        daily = daily[0]
        self.assertEqual(daily.timestamp, timestamp_from)
        self.assertEqual(daily.json_data["sample_value"], total)
        self.assertEqual(daily.json_data["target_value"], target_value)
        self.assertEqual(daily.json_data["next"], expected_next)

    def test_candle_cache_is_not_converted_to_daily_without_all_timestamps(self):
        timestamp_to = get_min_time(get_current_time(), value="1d")
        timestamp_from = get_previous_time(timestamp_to, value="1d")
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=timestamp_from,
            frequency=Frequency.HOUR,
            json_data={"sample_value": 0},
        )
        convert_candle_cache_to_daily(self.candle)
        candle_cache = CandleCache.objects.filter(candle=self.candle)
        self.assertFalse(candle_cache.filter(frequency=Frequency.DAY).exists())
        self.assertEqual(candle_cache.filter(frequency=Frequency.HOUR).count(), 1)

    def test_convert_candle_cache_to_daily_with_existing_daily_cache(self):
        timestamp_to = get_min_time(get_current_time(), value="1d")
        day_three_from = get_previous_time(timestamp_to, value="3d")
        day_two_from = get_previous_time(timestamp_to, value="2d")
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=timestamp_to,
            frequency=Frequency.DAY,
            json_data={"sample_value": 24},
        )
        for day_from in (day_three_from, day_two_from):
            for hour in range(24):
                CandleCache.objects.create(
                    candle=self.candle,
                    timestamp=day_from + pd.Timedelta(f"{hour}h"),
                    frequency=Frequency.HOUR,
                    json_data={"sample_value": hour + 1},
                )

        convert_candle_cache_to_daily(self.candle)

        candle_cache = CandleCache.objects.filter(candle=self.candle)
        self.assertEqual(candle_cache.filter(frequency=Frequency.DAY).count(), 3)
        self.assertFalse(
            candle_cache.filter(
                frequency=Frequency.HOUR,
                timestamp__gte=day_three_from,
                timestamp__lt=timestamp_to,
            ).exists()
        )
