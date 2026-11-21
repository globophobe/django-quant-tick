import quant_tick.models.checkpoints
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("quant_tick", "0010_checkpoints_and_trade_file_paths"),
    ]

    operations = [
        migrations.RenameField(
            model_name="checkpoint",
            old_name="kind",
            new_name="checkpoint_type",
        ),
        migrations.RenameField(
            model_name="checkpoint",
            old_name="state",
            new_name="json_data",
        ),
        migrations.AddField(
            model_name="checkpoint",
            name="file_data",
            field=models.FileField(
                blank=True,
                max_length=255,
                upload_to=quant_tick.models.checkpoints.checkpoint_upload_to,
            ),
        ),
        migrations.AlterField(
            model_name="checkpoint",
            name="checkpoint_type",
            field=models.CharField(
                choices=[
                    ("phoenix_positions", "Phoenix positions"),
                    ("phoenix_snapshot", "Phoenix snapshot"),
                    ("liquidation_exposure", "Liquidation exposure"),
                ],
                max_length=255,
            ),
        ),
    ]
