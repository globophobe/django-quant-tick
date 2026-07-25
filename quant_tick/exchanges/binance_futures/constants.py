from datetime import timedelta

API_URL = "https://fapi.binance.com/fapi/v1"
DATA_API_URL = "https://fapi.binance.com/futures/data"
S3_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
METRICS_S3_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

TRADE_MAX_RESULTS = 1000
MIN_ELAPSED_PER_REQUEST = 0
MARKET_HISTORY_MAX_RESULTS = 500
MARKET_HISTORY_INTERVAL = "5m"
MARKET_HISTORY_REST_RETENTION = timedelta(days=30)
