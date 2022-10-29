Example tick data
-----------------

| uid (1) | timestamp          | nanoseconds (2) | price (3) | volume (3) | notional (3) | tickRule (4) |
|---------|--------------------|-----------------|-----------|------------|--------------|--------------|
| ...     | ...00:27:17.367156 | 0               | 456.98    | 2000       | 4.3765591... | -1           |
| ...     | ...00:44:59.302471 | 0               | 457.1     | 1000       | 2.1877050... | 1            |
| ...     | ...00:44:59.302471 | 0               | 457.11    | 2000       | 4.3753144... | 1            |

Note:

1. UID
* Some exchanges use integer IDs, others UUIDs.

2. Nanoseconds
* Many databases don't support nanoseconds. Then again, many exchanges don't either. Timestamps are parsed with `pd.to_datetime(value)`, as `datetime.datetime` doesn't support nanoseconds. If nanoseconds exist, they are saved to the nanoseconds column.

3. Price, volume, and notional
* `django-quant-candles` stores decimals with `max_digits=76` and `decimal_places=38`, similar to BigQuery's [BIGNUMERIC](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types) type. Some exchange REST APIs return floats rather than strings, which can be an issue when parsing to `Decimal`. Such exchanges include Bitfinex, BitMEX, Bybit, and FTX. For those exchanges, response is loaded with `json.loads(response.read(), parse_float=Decimal))`.

4. Tick rule
* Plus ticks and zero plus ticks have tick rule 1, minus ticks and zero minus ticks have tick rule -1.
