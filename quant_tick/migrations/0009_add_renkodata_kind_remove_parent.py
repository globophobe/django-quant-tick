# Generated manually

from django.db import migrations, models

from quant_tick.constants import RenkoKind


def populate_kind_from_parent(apps, schema_editor):
    """Populate kind field based on parent field before dropping parent."""
    RenkoData = apps.get_model("quant_tick", "RenkoData")
    # parent_id is None = BODY, parent_id is set = WICK
    RenkoData.objects.filter(parent__isnull=True).update(kind=RenkoKind.BODY)
    RenkoData.objects.filter(parent__isnull=False).update(kind=RenkoKind.WICK)


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
        # Populate kind from parent
        migrations.RunPython(populate_kind_from_parent, migrations.RunPython.noop),
        # Remove parent field
        migrations.RemoveField(
            model_name="renkodata",
            name="parent",
        ),
    ]
