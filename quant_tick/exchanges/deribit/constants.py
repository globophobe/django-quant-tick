from datetime import timedelta

API_URL = "https://www.deribit.com/api/v2/public"
HISTORY_API_URL = "https://history.deribit.com/api/v2/public"
MIN_ELAPSED_PER_REQUEST = 0.05
CANDLE_MAX_RESULTS = 1000
TRADE_MAX_RESULTS = 1000
HISTORY_TRADE_MAX_RESULTS = 10000
FUNDING_FETCH_WINDOW = timedelta(days=30)
RECENT_TRADE_WINDOW = timedelta(hours=24)

CANDLE_RESOLUTIONS_BY_MINUTES = {
    1: "1",
    3: "3",
    5: "5",
    10: "10",
    15: "15",
    30: "30",
    60: "60",
    120: "120",
    180: "180",
    360: "360",
    720: "720",
    1440: "1D",
}
