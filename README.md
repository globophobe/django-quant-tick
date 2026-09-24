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

✅ Binance REST API, and [Binance Market Data](https://data.binance.vision/)

✅ Bitfinex REST API

✅ Bybit [S3](https://public.bybit.com/trading/)

✅ Coinbase REST API

✅ Deribit REST API

✅ Phoenix perpetuals REST API: historical fills, exchange candles, hourly funding, and open interest

Note: Exchanges without paginated REST APIs or an S3 repository, will never be supported.

Phoenix collection uses `exchange="phoenix"`, `symbol_type="perpetual"`, and the
market symbol, for example `BTC`. Set `date_from` to the market's trading start;
the earliest BTC fill returned by the public API was 2025-11-18 03:28:14 UTC.
Historical fills use cursor pagination and feed the existing trade aggregation
and validation path. Exchange candles include mark prices and support `8h`
through aggregation of native `4h` candles. Funding is stored as fractional
hourly rates. Perpetual statistics contain hourly open interest in base units
and its mark-price quote value, with mark/spot prices and the source slot retained
as metadata. Active symbols use the existing collection callbacks.

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

To use the scripts or deploy to Google Cloud, rename `.env.sample` to `.env`, and add the required settings.
