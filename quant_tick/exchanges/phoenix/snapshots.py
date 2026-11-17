"""Capture one validated account observation per current UTC hour.

These are forward-collected snapshots, not historical reconstructions. A row's
hour is the time the validated result became available, not the chain slot's
hour or the previous completed interval. Consumers use completed hourly rows;
the exact block time and availability time remain attached to the cohorts.

Only cohort fields are saved here. Native hourly OI and trade flows still have
to be fetched after the hour closes, and missing native data stays retryable.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
from django.db import transaction

from quant_tick.lib import get_current_time
from quant_tick.models import PerpetualStatsData

from ..quicknode.solana import available, rpc
from .accounts import cohort_values, decode_snapshot
from .events import PROGRAM

RATIO_FIELDS = (
    "long_account_ratio",
    "short_account_ratio",
    "long_short_account_ratio",
)


def fetch_snapshot(symbol):
    snapshot = decode_snapshot(
        rpc(
            "getProgramAccounts",
            [
                PROGRAM,
                {
                    "encoding": "base64+zstd",
                    "commitment": "finalized",
                    "withContext": True,
                },
            ],
        ),
        symbol.strip().upper(),
    )
    timestamp = rpc("getBlockTime", [snapshot["slot"]])
    if type(timestamp) is not int:
        raise ValueError("Invalid Phoenix snapshot block time")
    block_time = datetime.fromtimestamp(timestamp, UTC)
    values = cohort_values(snapshot)
    received = get_current_time()
    if not timedelta(0) <= received - block_time <= timedelta(minutes=5):
        raise ValueError("Phoenix snapshot is stale or has a future block time")
    return {
        **values,
        "cohort_timestamp": block_time,
        "cohort_available_at": received,
    }


def collect_snapshot(symbol, *, assert_lease_owned=None):
    if not available():
        return
    hour = pd.Timestamp(get_current_time()).floor("h").to_pydatetime()
    if PerpetualStatsData.objects.filter(
        symbol=symbol,
        frequency=60,
        timestamp=hour,
        json_data__cohort_snapshot_validated=True,
    ).exists():
        return
    values = fetch_snapshot(symbol.api_symbol)
    # A slow response can cross an hour boundary; never backdate its availability.
    hour = pd.Timestamp(values["cohort_available_at"]).floor("h").to_pydatetime()
    with transaction.atomic():
        if assert_lease_owned is not None:
            assert_lease_owned()
        row, _ = PerpetualStatsData.objects.select_for_update().get_or_create(
            symbol=symbol,
            frequency=60,
            timestamp=hour,
        )
        if (row.json_data or {}).get("cohort_snapshot_validated"):
            return
        for field in RATIO_FIELDS:
            setattr(row, field, values.pop(field))
        row.json_data = {**(row.json_data or {}), **values}
        row.save(update_fields=[*RATIO_FIELDS, "json_data"])
