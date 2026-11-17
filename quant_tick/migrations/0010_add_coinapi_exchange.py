import quant_tick.models.trades
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quant_tick', '0009_add_phoenix_exchange'),
    ]

    operations = [
        migrations.AlterField(
            model_name='symbol',
            name='exchange',
            field=models.CharField(choices=[('binance', 'Binance'), ('binance-futures', 'Binance Futures'), ('bitfinex', 'Bitfinex'), ('bitmex', 'BitMEX'), ('bybit', 'Bybit'), ('bybit-linear', 'Bybit Linear'), ('bybit-inverse', 'Bybit Inverse'), ('coinapi', 'CoinAPI'), ('coinbase', 'Coinbase'), ('deribit', 'Deribit'), ('hyperliquid', 'Hyperliquid'), ('phoenix', 'Phoenix')], max_length=255, verbose_name='exchange'),
        ),
        migrations.AlterField(
            model_name='taskstate',
            name='exchange',
            field=models.CharField(choices=[('all', 'All'), ('binance', 'Binance'), ('binance-futures', 'Binance Futures'), ('bitfinex', 'Bitfinex'), ('bitmex', 'BitMEX'), ('bybit', 'Bybit'), ('bybit-linear', 'Bybit Linear'), ('bybit-inverse', 'Bybit Inverse'), ('coinapi', 'CoinAPI'), ('coinbase', 'Coinbase'), ('deribit', 'Deribit'), ('hyperliquid', 'Hyperliquid'), ('phoenix', 'Phoenix')], max_length=255, verbose_name='exchange'),
        ),
        migrations.AlterField(
            model_name='websocketdata',
            name='exchange',
            field=models.CharField(choices=[('binance', 'Binance'), ('binance-futures', 'Binance Futures'), ('bitfinex', 'Bitfinex'), ('bitmex', 'BitMEX'), ('bybit', 'Bybit'), ('bybit-linear', 'Bybit Linear'), ('bybit-inverse', 'Bybit Inverse'), ('coinapi', 'CoinAPI'), ('coinbase', 'Coinbase'), ('deribit', 'Deribit'), ('hyperliquid', 'Hyperliquid'), ('phoenix', 'Phoenix')], max_length=255, verbose_name='exchange'),
        ),
        migrations.AlterField(
            model_name='tradedata',
            name='aggregated_data',
            field=models.FileField(blank=True, max_length=255, upload_to=quant_tick.models.trades.upload_aggregated_data_to, verbose_name='aggregated data'),
        ),
        migrations.AlterField(
            model_name='tradedata',
            name='filtered_data',
            field=models.FileField(blank=True, max_length=255, upload_to=quant_tick.models.trades.upload_filtered_data_to, verbose_name='filtered data'),
        ),
        migrations.AlterField(
            model_name='tradedata',
            name='raw_data',
            field=models.FileField(blank=True, max_length=255, upload_to=quant_tick.models.trades.upload_raw_data_to, verbose_name='raw data'),
        ),
    ]
