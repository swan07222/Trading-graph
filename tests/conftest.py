# tests/conftest.py
"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
import os
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset cache before each test"""
    from data.cache import get_cache
    cache = get_cache()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model saves"""
    from config.settings import CONFIG
    old_dir = CONFIG.model_dir
    CONFIG._model_dir_override = tmp_path
    yield tmp_path
    CONFIG._model_dir_override = None

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