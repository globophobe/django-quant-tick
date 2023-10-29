S3_URL = "https://public.bybit.com/trading/"

BYBIT_MAX_REQUESTS_RESET = "BYBIT_MAX_REQUESTS_RESET"
BYBIT_TOTAL_REQUESTS = "BYBIT_TOTAL_REQUESTS"

INVERSE_CONTRACTS = ("BTCUSD", "ETHUSD", "EOSUSD", "XRPUSD")

API_URL = "https://api.bybit.com/"

MAX_RESULTS = 200
MAX_REQUESTS = 50
# Bybit docs say "rate_limit_status", "rate_limit", and "rate_limit_reset_ms" are
# returned, but they are in neither response headers nor data
MAX_REQUESTS_RESET = 120  # 2 minutes
MIN_ELAPSED_PER_REQUEST = 0
