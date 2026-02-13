# tests/conftest.py
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset cache before each test"""
    if os.environ.get("TRADING_SKIP_TEST_CACHE_CLEAR", "0") == "1":
        yield
        return
    old_manual = os.environ.get("TRADING_MANUAL_CACHE_DELETE")
    os.environ["TRADING_MANUAL_CACHE_DELETE"] = "1"
    from data.cache import get_cache
    cache = get_cache()
    cache.clear()
    yield
    cache.clear()
    if old_manual is None:
        os.environ.pop("TRADING_MANUAL_CACHE_DELETE", None)
    else:
        os.environ["TRADING_MANUAL_CACHE_DELETE"] = old_manual

@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model saves (cache-safe)."""
    from config.settings import CONFIG
    old = str(CONFIG.model_dir)
    CONFIG.set_model_dir(str(tmp_path))
    yield tmp_path
    CONFIG.set_model_dir("")  

@pytest.fixture(autouse=True, scope="session")
def force_offline_for_tests():
    """
    Make pytest deterministic and fast:
    - Avoid network calls (AkShare/Yahoo)
    - Allow tests to run even without data sources
    """
    old = os.environ.get("TRADING_OFFLINE")
    os.environ["TRADING_OFFLINE"] = "1"
    yield
    if old is None:
        os.environ.pop("TRADING_OFFLINE", None)
    else:
        os.environ["TRADING_OFFLINE"] = old
