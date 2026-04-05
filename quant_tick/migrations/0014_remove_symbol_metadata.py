from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("quant_tick", "0013_taskstate"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="symbol",
            name="global_symbol",
        ),
        migrations.RemoveField(
            model_name="symbol",
            name="symbol_type",
        ),
        migrations.RemoveField(
            model_name="symbol",
            name="recent_error_at",
        ),
        migrations.DeleteModel(
            name="GlobalSymbol",
        ),
    ]
