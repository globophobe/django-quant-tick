# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("quant_tick", "0008_remove_candle_symbols_remove_strategy_symbol_and_more"),
    ]

    operations = [
        # Add kind field with default
        migrations.AddField(
            model_name="renkodata",
            name="kind",
            field=models.CharField(
                choices=[("body", "body"), ("wick", "wick")],
                db_index=True,
                default="body",
                max_length=10,
                verbose_name="kind",
            ),
        ),
        migrations.RemoveField(
            model_name="renkodata",
            name="parent",
        ),
    ]
