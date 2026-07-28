API_URL = "https://api.bybit.com"
S3_URL = "https://public.bybit.com/trading"

MIN_ELAPSED_PER_REQUEST = 0.05
CANDLE_MAX_RESULTS = 1000
FUNDING_MAX_RESULTS = 200
MARKET_HISTORY_MAX_RESULTS = 200
MARKET_HISTORY_INTERVAL = "4h"

CANDLE_RESOLUTIONS_BY_MINUTES = {
    1: "1",
    3: "3",
    5: "5",
    15: "15",
    30: "30",
    60: "60",
    120: "120",
    240: "240",
    360: "360",
    720: "720",
    1440: "D",
    10080: "W",
}
