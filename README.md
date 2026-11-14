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

✅ CoinAPI spot and Hyperliquid perpetual trades and OHLCV REST API (API key required)

✅ Coinbase REST API

✅ Deribit REST API

✅ Optional CoinAPI Hyperliquid backfill and Femtos positioning history (API keys required)

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

Hyperliquid data sources
------------------------

Configure one `hyperliquid` perpetual symbol with the native coin, such as `BTC`.
Native Hyperliquid is preferred for trades, candles and funding. Optional keys
in the process environment or `demo/.env` enable additional sources on that
same symbol:

- `COINAPI_API_KEY`: historical trades and candle gaps for standard USDC
  perpetuals. Validated native WebSocket minutes are retained, including on
  repair retries; only uncovered intervals use CoinAPI. Existing stored periods
  are skipped by the normal collection path. HIP-3 and spot vendor mappings are
  not enabled. Candle fallback supports native intervals from one minute through
  one day; longer calendar bars remain native.
- `FEMTOS_API_KEY`: hourly open interest, opening/closing flows, account ratios
  and position-size cohorts through `perpetual_stats`. Funding stays native.
  Without the key, scheduled collection skips this optional history.

Supplying a CoinAPI key enables paid fallback during ordinary collection. Its
[trade history](https://www.coinapi.io/products/market-data-api/docs/rest-api/trades/trades/symbol_id/history/get)
and [OHLCV history](https://www.coinapi.io/products/market-data-api/docs/rest-api/ohlcv/ohlcv/symbol_id/history/get)
responses consume credits per 100 returned items. Start with one completed
minute. Public coverage metadata is cached daily to avoid requests before the
provider's advertised history begins. Native candles cover only the most recent
5,000 bars at each interval; older intervals can use CoinAPI.

Native and vendor trades never overlap within a minute. Parquet rows retain
`data_source`, and trade partitions summarize it in `json_data.data_sources`.
Trade aggregation and filtering preserve source metadata. Candle rows retain
their source in `json_data`; positioning rows use `market_history_source`.
All data remains under the Hyperliquid symbol and storage path, forming one
venue input for adaptive candles and research.

From the repository root, prepare the local database and open a shell:

```sh
cd demo
DJANGO_SETTINGS_MODULE=demo.settings.development.local ../.venv/bin/python manage.py migrate
DJANGO_SETTINGS_MODULE=demo.settings.development.local ../.venv/bin/python manage.py shell
```

Then collect one minute into an inactive Hyperliquid symbol:

```python
from datetime import UTC, datetime, timedelta

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import api
from quant_tick.models import Symbol, TradeData

start = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(days=1)
symbol, _ = Symbol.objects.get_or_create(
    exchange=Exchange.HYPERLIQUID,
    api_symbol="BTC",
    symbol_type=SymbolType.PERPETUAL,
    defaults={"save_raw": True, "save_aggregated": True, "is_active": False,
              "date_from": start.date()},
)
api(symbol, start, start + timedelta(minutes=1))
list(TradeData.objects.filter(symbol=symbol).values("timestamp", "ok"))
```

Trade `size` and candle `volume_traded` map to the base-quantity field `notional`.
Trade quote-value `volume` is `price * size`; OHLCV does not supply quote turnover.
Raw files retain `coinapi_uuid`, `time_coinapi`, and the exchange `id_trade` when
provided. Trade direction must be reported `BUY` or `SELL`: estimated and unknown
sides abort the slice before writing, rather than contaminating flow features.

`ok=True` means collected trades agree with the available minute candles,
preferring native candles. Agreement with CoinAPI OHLCV alone is not independent
evidence of venue coverage.
Missing validation candles remain indeterminate, and quantity mismatches remain
failed validation. A mismatch can reflect inconsistent vendor datasets rather
than missing trades. Compare native candles where available before retrying a
paid range.

Absent keys leave native collection available. Provider authentication,
exhausted-credit and response errors propagate; they are not treated as empty
history or as a reason to replace usable native data. Empty trade responses
without zero-quantity validation candles leave the gap unfilled for the next
pass. Native API failures do not trigger paid requests. In the scheduled
exchange-data workflow, native funding and candles run before optional
positioning collection.

The separate `coinapi` symbol adapter remains available for provider comparisons,
including spot markets such as `COINBASE_SPOT_BTC_USD`. It uses full CoinAPI IDs
and separate storage; do not add that comparison as another Hyperliquid venue
input. The separate `HYPERLIQUIDL4` feed and other derivatives are not enabled.

Femtos Hyperliquid history
-------------------------

Femtos is a positioning provider for the same `hyperliquid` perpetual symbol,
including native HIP-3 names such as `xyz:TSLA`. A separate Femtos exchange or
symbol is not needed.

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

From the local Django shell described above, collect one completed hour:

```python
from datetime import UTC, datetime, timedelta
from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.perpetual_stats import perpetual_stats
from quant_tick.models import Symbol

end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
start = end - timedelta(hours=1)
symbol, _ = Symbol.objects.get_or_create(
    exchange=Exchange.HYPERLIQUID, api_symbol="BTC", symbol_type=SymbolType.PERPETUAL,
    defaults={"is_active": False, "date_from": start.date()},
)
perpetual_stats(symbol, start, end)
```

`export_perpetual_stats` includes Femtos account ratios and a JSON-text
`json_data` column by default, preserving exact cohort/flow quantities and
timestamps. Other sources can opt into that metadata with `--field json_data`.
Requests are split into at most 30 days and 720 observations to avoid heavy
history calls. Authentication and exhausted-credit errors stop collection.

Historical Hyperliquid trades are **not** supported by Femtos's REST endpoint:
its time filters and cursor history are for EVM swaps, while HyperCore supplies
only recent trades. Trade backfill uses CoinAPI; Femtos positioning history
does not reconstruct trade-level flow.
