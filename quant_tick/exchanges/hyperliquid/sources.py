"""Optional historical providers keep the native Hyperliquid market identity."""

import os


def femtos_available():
    return bool(os.environ.get("FEMTOS_API_KEY", "").strip())
