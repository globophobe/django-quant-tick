import os
import subprocess
import sys
from pathlib import Path

from django.test import SimpleTestCase


class TestEnvironmentTest(SimpleTestCase):
    def test_settings_ignore_dotenv_and_inherited_credentials(self):
        script = """
import importlib
import os
import sys
from types import ModuleType
from unittest.mock import patch

entry_point = sys.argv[1]
sys.argv = ["manage.py", "test"] if entry_point == "django" else ["embedded"]
if entry_point == "pytest":
    sys.modules["pytest"] = ModuleType("pytest")
with patch("dotenv.main.DotEnv.dict", side_effect=AssertionError("dotenv read")):
    settings = importlib.import_module("demo.settings.development.local")
    with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "0"}):
        import tasks
    from dotenv import load_dotenv
    assert not load_dotenv()

assert settings.SECRET_KEY == "development-secret-key"
for name in (
    "QUICKNODE_API_KEY", "QUICKNODE_HYPERCORE_URL", "QUICKNODE_SOLANA_URL",
    "BINANCE_API_KEY", "SENTRY_DSN", "GOOGLE_APPLICATION_CREDENTIALS",
):
    assert not os.environ.get(name), name
import django
django.setup()
from quant_tick.exchanges.quicknode.api import available
assert not available()
with patch.dict(os.environ, {"QUICKNODE_API_KEY": "mock-key"}):
    assert available()
"""
        environment = {
            **os.environ,
            "DJANGO_SETTINGS_MODULE": "demo.settings.development.local",
            "PYTHON_DOTENV_DISABLED": "0",
            **dict.fromkeys(
                (
                    "SECRET_KEY",
                    "QUICKNODE_API_KEY",
                    "QUICKNODE_HYPERCORE_URL",
                    "QUICKNODE_SOLANA_URL",
                    "BINANCE_API_KEY",
                    "SENTRY_DSN",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                ),
                "inherited-test-sentinel",
            ),
        }
        for entry_point in ("django", "pytest"):
            with self.subTest(entry_point=entry_point):
                result = subprocess.run(
                    [sys.executable, "-c", script, entry_point],
                    cwd=Path(__file__).resolve().parents[2] / "demo",
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
