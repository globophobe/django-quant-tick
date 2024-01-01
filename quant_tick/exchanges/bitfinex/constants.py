BITFINEX_MAX_REQUESTS_RESET = "BITFINEX_MAX_REQUESTS_RESET"
BITFINEX_TOTAL_REQUESTS = "BITFINEX_TOTAL_REQUESTS"

API_URL = "https://api-pub.bitfinex.com/v2"
MAX_REQUESTS = (
    10  # Specified as 90 for trades and candles in the docs, but was rate-limited...
)
MAX_REQUESTS_RESET = 60  # 1 minute
API_MAX_RESULTS = 10000  # Actual API max
MAX_RESULTS = 5000  # Bitfinex API is actually 10000, but 5000 is sufficient
MIN_ELAPSED_PER_REQUEST = 0
