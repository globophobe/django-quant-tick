import quant_tick.models.trades
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quant_tick', '0010_add_coinapi_exchange'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tradedata',
            name='aggregated_data',
            field=models.FileField(blank=True, max_length=1024, upload_to=quant_tick.models.trades.upload_aggregated_data_to, verbose_name='aggregated data'),
        ),
        migrations.AlterField(
            model_name='tradedata',
            name='filtered_data',
            field=models.FileField(blank=True, max_length=1024, upload_to=quant_tick.models.trades.upload_filtered_data_to, verbose_name='filtered data'),
        ),
        migrations.AlterField(
            model_name='tradedata',
            name='raw_data',
            field=models.FileField(blank=True, max_length=1024, upload_to=quant_tick.models.trades.upload_raw_data_to, verbose_name='raw data'),
        ),
    ]
