# What?

Django Quant Tick aggregates candlesticks from high frequency tick data.

# Why?

Tick data is preferable for analyzing financial markets. Candlesticks aggregated by `django-quant-tick` are equally informationally dense. Such candles can be useful for analyzing financial markets. As an example, refer to ["Low-Frequency Traders in a High-Frequency World: A Survival Guide"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2150876) and ["The Volume Clock: Insights into the High Frequency Paradigm"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2034858). Lopez de Prado recommends volume candlesticks, however they are are computationally expensive to generate.

Tick data may be downloaded raw, or optionally aggregated. Aggregation can save disk space, and may increase information. There are 2 complementary aggregations. The first is by equal symbol, timestamp, nanoseconds and tick rule. The second is by filtering significant trades, for example at least $1000.

By aggregating and filtering raw tick data, volume candlesticks can be computed faster, with little loss in precision.

1. First tick data may be aggregated by equal symbol, timestamp, nanoseconds and tick rule. Aggregating trades in this way can increase information, as they are either orders of size or stop loss cascades. As well, the number of rows can be reduced by 30-50%

2. By filtering aggregated rows, for example only writing a row when an aggregated trade is greater than `significant_trade_filter >= 1000`, the number of rows can be reduced more.

3. Clustering trades by trade direction, such that a row is created only if the tick rule changes, may further reduce the number of rows.


# How?

Whenever possible, data is downloaded from the exchange's AWS S3 repositories. Otherwise, it is downloaded using their REST APIs. 

A database, preferably PostgreSQL, is required. Data is saved to the database after aggregation and filtering. 

Candles are aggregated at 1 minute intervals, and validated with the exchange's historical candle API.

[Notes](https://github.com/globophobe/django-quant-tick/blob/main/NOTES.md).

Supported exchanges
-------------------

✅ Aster perpetuals REST API and funding history

✅ Binance REST API, and [Binance Market Data](https://data.binance.vision/)

✅ Binance.US REST API

✅ Bitfinex REST API

✅ Bybit [S3](https://public.bybit.com/trading/)

✅ CoinAPI spot trades and OHLCV REST API (API key required)

✅ Coinbase REST API

✅ Deribit REST API

✅ Femtos Hyperliquid perpetual candles, funding and positioning history (API key required)

✅ OKX spot and linear perpetuals REST API and daily archives

✅ Phoenix perpetuals REST API

✅ Upbit REST API and daily archives

Note: Exchanges without paginated REST APIs or an S3 repository, will never be supported.

Installation
------------

For convenience, `django-quant-tick` can be installed from PyPI:

```
pip install django-quant-tick
```

Deployment
----------

For deployment, there are Dockerfiles. As well, there are invoke tasks for deployment to Google Cloud Run. Just as easily, the demo could be deployed to a VPS or AWS.

If using Google Cloud, it is recommended to use the Cloud SQL Auth proxy, and run the management commands to collect data from your local machine. Django Quant Tick will upload the trade data to the cloud.

```
cd demo
invoke start-proxy
python proxy.py trades
```

Then, configure a Cloud Workflow to collect data in the cloud. There is an example workflow in the [invoke tasks](https://github.com/globophobe/django-quant-tick/blob/main/demo/tasks.py).

Environment
-----------

Copy `demo/env.sample` to `demo/.env` if it does not already exist, then add the required settings.

CoinAPI trial
-------------

Set `COINAPI_API_KEY` in the process environment or `demo/.env`. The adapter uses
the documented [trade history](https://www.coinapi.io/products/market-data-api/docs/rest-api/trades/trades/symbol_id/history/get)
and [OHLCV history](https://www.coinapi.io/products/market-data-api/docs/rest-api/ohlcv/ohlcv/symbol_id/history/get)
endpoints. Start with one completed minute: historical responses consume credits
per 100 returned items, and a busy minute can require several pages. An API key
with the appropriate historical-data entitlement is needed; no live trial has
been validated yet.

From the repository root, prepare the local database and open a shell:

```sh
cd demo
DJANGO_SETTINGS_MODULE=demo.settings.development.local ../.venv/bin/python manage.py migrate
DJANGO_SETTINGS_MODULE=demo.settings.development.local ../.venv/bin/python manage.py shell
```

Then collect one minute into a separate, inactive comparison symbol:

```python
from datetime import UTC, datetime, timedelta

from quant_tick.constants import Exchange
from quant_tick.exchanges.api import api
from quant_tick.models import Symbol, TradeData

start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(days=1)
symbol, _ = Symbol.objects.get_or_create(
    exchange=Exchange.COINAPI,
    api_symbol="BINANCE_SPOT_BTC_USDT",
    defaults={"save_raw": True, "save_aggregated": True, "is_active": False,
              "date_from": start.date()},
)
api(symbol, start, start + timedelta(minutes=1))
list(TradeData.objects.filter(symbol=symbol).values("timestamp", "ok"))
```

Use the full CoinAPI `symbol_id`, not the exchange-native pair. Only spot markets
are supported initially. The `coinapi` source keeps storage separate from native
exchange symbols. Existing trade aggregation and adaptive-candle paths apply.

Trade `size` and candle `volume_traded` map to the base-quantity field `notional`.
Trade quote-value `volume` is `price * size`; OHLCV does not supply quote turnover.
Raw files retain `coinapi_uuid`, `time_coinapi`, and the exchange `id_trade` when
provided. Trade direction must be reported `BUY` or `SELL`: estimated and unknown
sides abort the slice before writing, rather than contaminating flow features.

`ok=True` means collected trades agree with CoinAPI's minute quantities. Compare
the same venue, pair and UTC window with direct exchange data separately;
CoinAPI trades and CoinAPI OHLCV are not independent evidence of venue coverage.
Missing validation candles remain indeterminate, and quantity mismatches remain
failed validation.

Femtos Hyperliquid history
-------------------------

Set `FEMTOS_API_KEY` in the process environment or `demo/.env`. Use the `femtos`
source with `symbol_type="perpetual"` and a native coin such as `BTC` or
`xyz:TSLA`. This keeps vendor observations separate from the direct Hyperliquid
feed. Supported collectors are `exchange_candles`, `funding` and
`perpetual_stats`, including the normal scheduled exchange-data workflow.

The [candle API](https://femtos.xyz/reference/candles) supplies minute through
daily OHLCV: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h and 1d. Base quantity
maps to `notional`; quote turnover maps to `volume`. Gaps remain absent and
unknown quote turnover remains null, including native Hyperliquid candles the
provider substitutes over gaps in its own fills. Neither missing nor forming
bars are filled locally.

Hourly [open-interest history](https://femtos.xyz/reference/open-interest)
starts in May 2023; opening/closing flows become available in March 2025 and
[size cohorts](https://femtos.xyz/reference/open-interest-cohorts) in September
2025. Availability can vary by market and archive gaps. Open interest counts
longs and shorts together; `single_open_interest` is half that quantity. OI and
flow fields describe the completed hour. Account ratios use the cohort at its
start, with the precise funding-block time retained as `cohort_timestamp`.
Full cohort tiers, top-ten sizes, raw account counts and flows stay in the row's
`json_data`; large positions are not labelled as Binance-style top traders.
An hour missing its expected cohort stays retryable. Older OI-only history is
retained; unavailable flows stay null.

Funding is labelled by settlement time, one hour after the provider's OI bar
label. Collection starts June 8, 2023; earlier rates are the provider's hourly
distribution of eight-hour settlements and are excluded from funding events.

From the local Django shell described above, collect one completed hour:

```python
from datetime import UTC, datetime, timedelta
from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import exchange_candles, funding
from quant_tick.exchanges.perpetual_stats import perpetual_stats
from quant_tick.models import Symbol

end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
start = end - timedelta(hours=1)
symbol, _ = Symbol.objects.get_or_create(
    exchange=Exchange.FEMTOS, api_symbol="BTC",
    defaults={"symbol_type": SymbolType.PERPETUAL, "is_active": False,
              "exchange_candle_resolution": "1h", "date_from": start.date()},
)
exchange_candles(symbol, start, end)
funding(symbol, start, end)
perpetual_stats(symbol, start, end)
```

`export_perpetual_stats` includes Femtos account ratios and a JSON-text
`json_data` column by default, preserving exact cohort/flow quantities and
timestamps. Other sources can opt into that metadata with `--field json_data`.
Requests are split into at most 30 days and 720 observations to avoid heavy
history calls. Authentication and exhausted-credit errors stop collection.

Historical Hyperliquid trades are **not** supported by Femtos's REST endpoint:
its time filters and cursor history are for EVM swaps, while HyperCore supplies
only recent trades. Keep the direct `hyperliquid` collector for trade history
and adaptive candles; Femtos OHLCV cannot reconstruct trade-level flow.
