from __future__ import annotations

import os
import sys
from pathlib import Path

TEST_CREDENTIALS = (
    "BINANCE_API_KEY",
    "QUICKNODE_API_KEY",
    "QUICKNODE_HYPERCORE_URL",
    "QUICKNODE_SOLANA_URL",
    "SECRET_KEY",
    "SENTRY_DSN",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def configure_test_environment() -> None:
    """Disable dotenv and inherited credentials; tests supply their own fakes."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    for name in TEST_CREDENTIALS:
        os.environ.pop(name, None)


def is_test() -> bool:
    """Return True when running under Django's test command or pytest."""
    if "pytest" in sys.modules:
        return True
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        return True
    command = Path(sys.argv[0]).name
    if command == "pytest":
        return True
    return len(sys.argv) > 2 and sys.argv[1] == "-m" and sys.argv[2] == "pytest"
