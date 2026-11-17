from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quant_tick', '0010_add_coinapi_exchange'),
    ]

    operations = [
        migrations.AlterField(
            model_name='symbol',
            name='exchange',
            field=models.CharField(choices=[('binance', 'Binance'), ('binance-futures', 'Binance Futures'), ('bitfinex', 'Bitfinex'), ('bitmex', 'BitMEX'), ('bybit', 'Bybit'), ('bybit-linear', 'Bybit Linear'), ('bybit-inverse', 'Bybit Inverse'), ('coinbase', 'Coinbase'), ('deribit', 'Deribit'), ('hyperliquid', 'Hyperliquid'), ('phoenix', 'Phoenix')], max_length=255, verbose_name='exchange'),
        ),
        migrations.AlterField(
            model_name='taskstate',
            name='exchange',
            field=models.CharField(choices=[('all', 'All'), ('binance', 'Binance'), ('binance-futures', 'Binance Futures'), ('bitfinex', 'Bitfinex'), ('bitmex', 'BitMEX'), ('bybit', 'Bybit'), ('bybit-linear', 'Bybit Linear'), ('bybit-inverse', 'Bybit Inverse'), ('coinbase', 'Coinbase'), ('deribit', 'Deribit'), ('hyperliquid', 'Hyperliquid'), ('phoenix', 'Phoenix')], max_length=255, verbose_name='exchange'),
        ),
        migrations.AlterField(
            model_name='websocketdata',
            name='exchange',
            field=models.CharField(choices=[('binance', 'Binance'), ('binance-futures', 'Binance Futures'), ('bitfinex', 'Bitfinex'), ('bitmex', 'BitMEX'), ('bybit', 'Bybit'), ('bybit-linear', 'Bybit Linear'), ('bybit-inverse', 'Bybit Inverse'), ('coinbase', 'Coinbase'), ('deribit', 'Deribit'), ('hyperliquid', 'Hyperliquid'), ('phoenix', 'Phoenix')], max_length=255, verbose_name='exchange'),
        ),
    ]
