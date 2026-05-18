from __future__ import annotations

import sys
from pathlib import Path


def is_test() -> bool:
    """Return True when running under Django's test command or pytest."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        return True
    command = Path(sys.argv[0]).name
    if command == "pytest":
        return True
    return len(sys.argv) > 2 and sys.argv[1] == "-m" and sys.argv[2] == "pytest"
