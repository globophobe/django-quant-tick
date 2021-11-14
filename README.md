# What?

Django Cryptofeed Werks aggregates candlesticks using tick data from financial exchanges. 

Raw ticks are aggregated if equal symbol, timestamp, nanoseconds and tick rule. As described in the accompanying project [cryptofeed-werks](https://github.com/globophobe/cryptofeed-werks), aggregating trades in this way can increase information, as they are either orders of size or stop loss cascades.

As well, the number of rows can be reduced by 30-50%

By filtering aggregated rows, for example only writing a row when an aggregated trade is greater than `min_volume >= 1000`, the number of rows can be reduced more.


# Why?

Candlesticks aggregated by `django-cryptofeed-werks` are informationally dense. Such data can be useful for analyzing financial markets. As an example, refer to 
["Low-Frequency Traders in a High-Frequency World: A Survival Guide"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2150876) and ["The Volume Clock: Insights into the High Frequency Paradigm"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2034858). Lopez de Prado recommends volume bars, however they are are computationally expensive to generate. 

By filtering and aggregating trades, they can be computed faster, with little loss in precision.

# How?

Whenever possible, data is downloaded from the exchange's AWS S3 repositories. Otherwise, it is downloaded using their REST APIs. 

A database, preferably PostgreSQL, is required. Data is saved to the database after aggregation and filtering. 

Candles are aggregated at 1 minute intervals, and validated with the exchange's historical candle API.

[Notes](https://github.com/globophobe/django-cryptofeed-werks/blob/main/NOTES.md).


Supported exchanges
-------------------

:white_medium_square: Alpaca Markets

:white_check_mark: Binance REST API (requires API key)

:white_check_mark: Bitfinex REST API

:white_medium_square: Bitflyer REST API

:white_check_mark: BitMEX REST API, and [S3](https://public.bitmex.com/) repository

:white_check_mark: Bybit REST API, and [S3](https://public.bybit.com/) repository

:white_check_mark: Coinbase Pro REST API

:white_medium_square: Deribit REST API

:white_check_mark: FTX REST API

Note: Exchanges without paginated REST APIs will never be supported.

For deployment, there are Dockerfiles. As well there are invoke tasks for rapid deployment to Google Cloud Run.


Installation
------------

For convenience, `django-cryptofeed-werks` can be installed from PyPI:

```
pip install django-cryptofeed-werks
```

Environment
-----------

To use the scripts or deploy to GCP, rename `.env.sample` to `.env`, and add the required settings.


Future Plans
------------

Allow use as an aggr historical data source as decribed by [aggr](https://github.com/Tucsky/aggr#implement-historical-data).
