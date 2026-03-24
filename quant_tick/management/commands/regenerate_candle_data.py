import pandas as pd

from quant_tick.lib import aggregate_candle, aggregate_candles
from quant_tick.management.base import BaseTradeDataCommand
from quant_tick.models import TradeData


class Command(BaseTradeDataCommand):
    """Regenerate candle_data from filtered_data."""

    help = "Regenerate TradeData.candle_data from filtered_data."

    def handle(self, *args, **options) -> None:
        """Run command."""
        for kwargs in super().handle(*args, **options):
            qs = TradeData.objects.filter(
                symbol=kwargs["symbol"],
                timestamp__gte=kwargs["timestamp_from"],
                timestamp__lt=kwargs["timestamp_to"],
            ).exclude(filtered_data="").exclude(candle_data="")
            total = qs.count()
            self.stdout.write(f"{kwargs['symbol']}: {total} records")
            for i, obj in enumerate(qs.iterator(chunk_size=100), 1):
                filtered = obj.get_data_frame("filtered_data")
                if filtered is None or not len(filtered):
                    continue
                ts_from = obj.timestamp
                ts_to = ts_from + pd.Timedelta(f"{obj.frequency}min")
                # Read old candle_data for validated + exchange* columns.
                old_candles = obj.get_data_frame("candle_data")
                # Regenerate from filtered_data.
                new_candles = aggregate_candles(filtered, ts_from, ts_to)
                # Copy validation columns and match original column order.
                if old_candles is not None and len(old_candles) and len(new_candles):
                    validation_cols = [
                        c
                        for c in old_candles.columns
                        if c in ("validated", "exchangeNotional", "exchangeVolume")
                    ]
                    for col in validation_cols:
                        new_candles[col] = old_candles[col].reindex(new_candles.index)
                    new_candles = new_candles[old_candles.columns]
                # Regenerate json_data candle.
                obj.json_data = {"candle": aggregate_candle(filtered)}
                # Save.
                obj.candle_data.delete(save=False)
                obj.candle_data = TradeData.prepare_data(new_candles)
                obj.save()
                if i % 1000 == 0:
                    self.stdout.write(f"  {i}/{total}")
            self.stdout.write(f"{kwargs['symbol']}: done.")
