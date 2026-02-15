import datetime

BINANCE_API_KEY = "BINANCE_API_KEY"
BINANCE_MAX_WEIGHT = "BINANCE_MAX_WEIGHT"

SPOT_API_URL = "https://api.binance.com/api/v3"
FUTURES_API_URL = "https://fapi.binance.com"
S3_URL = "https://data.binance.vision/data/spot/daily/trades"
MAX_RESULTS = 1000

# Response 429, when x-mbx-used-weight-1m is 1200
MAX_WEIGHT = 1200
MIN_ELAPSED_PER_REQUEST = 0

# Switched to microseconds
MICROSECONDS_CUTOFF = datetime.date(2025, 1, 1)
