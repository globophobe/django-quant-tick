from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quant_tick', '0010_add_market_data_exchanges'),
    ]

    operations = [
        migrations.AlterField(
            model_name='symbol',
            name='significant_trade_filter',
            field=models.PositiveIntegerField(default=0, help_text='All trades less than the value of the significant trade filter will be aggregated, according to the the algorithm described in the documentation.', verbose_name='significant trade filter'),
        ),
    ]
