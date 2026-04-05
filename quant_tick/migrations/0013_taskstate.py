from django.db import migrations, models

import quant_tick.models.base

PAYLOAD_FIELD_MAP = (
    ("open", "open"),
    ("high", "high"),
    ("low", "low"),
    ("close", "close"),
    ("volume", "volume"),
    ("buyVolume", "buy_volume"),
    ("notional", "notional"),
    ("buyNotional", "buy_notional"),
    ("ticks", "ticks"),
    ("buyTicks", "buy_ticks"),
    ("realizedVariance", "realized_variance"),
    ("incomplete", "incomplete"),
)

DROP_KEYS = {
    "timestamp",
    "distribution",
    "roundEitherVolume",
    "roundEitherBuyVolume",
    "roundEitherNotional",
    "roundEitherBuyNotional",
}

UPDATE_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "buy_volume",
    "notional",
    "buy_notional",
    "ticks",
    "buy_ticks",
    "realized_variance",
    "incomplete",
    "extra_data",
]


def backfill_candledata_columns(apps, schema_editor):
    CandleData = apps.get_model("quant_tick", "CandleData")
    batch = []

    for obj in CandleData.objects.all().iterator(chunk_size=500):
        payload = dict(obj.json_data or {})

        for key, field_name in PAYLOAD_FIELD_MAP:
            if key in payload:
                setattr(obj, field_name, payload.pop(key))

        for key in DROP_KEYS:
            payload.pop(key, None)

        obj.extra_data = payload
        batch.append(obj)

        if len(batch) >= 500:
            CandleData.objects.bulk_update(batch, UPDATE_FIELDS)
            batch = []

    if batch:
        CandleData.objects.bulk_update(batch, UPDATE_FIELDS)


class Migration(migrations.Migration):
    dependencies = [
        ("quant_tick", "0012_alter_fundingrate_unique_together_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="TaskState",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "task_type",
                    models.CharField(
                        choices=[
                            ("aggregate_trades", "Aggregate trades"),
                            ("aggregate_candles", "Aggregate candles"),
                        ],
                        max_length=255,
                        verbose_name="task type",
                    ),
                ),
                (
                    "exchange",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("binance", "Binance"),
                            ("bitfinex", "Bitfinex"),
                            ("bitmex", "BitMEX"),
                            ("coinbase", "Coinbase"),
                            ("drift", "Drift"),
                            ("hyperliquid", "Hyperliquid"),
                        ],
                        default="",
                        max_length=255,
                        verbose_name="exchange",
                    ),
                ),
                (
                    "recent_error_at",
                    models.DateTimeField(
                        blank=True,
                        help_text="Last task error time used for observability.",
                        null=True,
                        verbose_name="recent error",
                    ),
                ),
                (
                    "recent_error_count",
                    models.PositiveIntegerField(
                        default=0,
                        help_text="Consecutive task errors used to determine exponential backoff.",
                        verbose_name="recent error count",
                    ),
                ),
                (
                    "next_fetch_at",
                    models.DateTimeField(
                        blank=True,
                        help_text="Do not run this task before this time if backoff is active.",
                        null=True,
                        verbose_name="next fetch at",
                    ),
                ),
                (
                    "locked_until",
                    models.DateTimeField(
                        blank=True,
                        help_text="Lease expiry for preventing overlapping task runs.",
                        null=True,
                        verbose_name="locked until",
                    ),
                ),
            ],
            options={
                "verbose_name": "task state",
                "verbose_name_plural": "task states",
                "db_table": "quant_tick_task_state",
                "ordering": ("task_type", "exchange"),
                "constraints": [
                    models.UniqueConstraint(
                        fields=("task_type", "exchange"),
                        name="quant_tick_task_state_unique",
                    )
                ],
            },
        ),
        migrations.AddField(
            model_name="candledata",
            name="open",
            field=models.DecimalField(
                blank=True,
                decimal_places=38,
                max_digits=76,
                null=True,
                verbose_name="open",
            ),
        ),
        migrations.AddField(
            model_name="candledata",
            name="high",
            field=models.DecimalField(
                blank=True,
                decimal_places=38,
                max_digits=76,
                null=True,
                verbose_name="high",
            ),
        ),
        migrations.AddField(
            model_name="candledata",
            name="low",
            field=models.DecimalField(
                blank=True,
                decimal_places=38,
                max_digits=76,
                null=True,
                verbose_name="low",
            ),
        ),
        migrations.AddField(
            model_name="candledata",
            name="close",
            field=models.DecimalField(
                blank=True,
                decimal_places=38,
                max_digits=76,
                null=True,
                verbose_name="close",
            ),
        ),
        migrations.AddField(
            model_name="candledata",
            name="volume",
            field=models.DecimalField(
                decimal_places=38,
                default=0,
                max_digits=76,
                verbose_name="volume",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="candledata",
            name="buy_volume",
            field=models.DecimalField(
                decimal_places=38,
                default=0,
                max_digits=76,
                verbose_name="buy volume",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="candledata",
            name="notional",
            field=models.DecimalField(
                decimal_places=38,
                default=0,
                max_digits=76,
                verbose_name="notional",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="candledata",
            name="buy_notional",
            field=models.DecimalField(
                decimal_places=38,
                default=0,
                max_digits=76,
                verbose_name="buy notional",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="candledata",
            name="ticks",
            field=models.PositiveIntegerField(default=0, verbose_name="ticks"),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="candledata",
            name="buy_ticks",
            field=models.PositiveIntegerField(default=0, verbose_name="buy ticks"),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="candledata",
            name="realized_variance",
            field=models.DecimalField(
                decimal_places=38,
                default=0,
                max_digits=76,
                verbose_name="realized variance",
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="candledata",
            name="incomplete",
            field=models.BooleanField(default=False, verbose_name="incomplete"),
        ),
        migrations.AddField(
            model_name="candledata",
            name="extra_data",
            field=models.JSONField(
                decoder=quant_tick.models.base.QuantTickDecoder,
                default=dict,
                encoder=quant_tick.models.base.QuantTickEncoder,
                verbose_name="extra data",
            ),
        ),
        migrations.RunPython(backfill_candledata_columns, migrations.RunPython.noop),
        migrations.RemoveField(
            model_name="candledata",
            name="json_data",
        ),
    ]
