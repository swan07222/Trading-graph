# tests/conftest.py
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _warmup_torch_import() -> None:
    """Import torch once at startup to stabilize DLL load order on Windows."""
    try:
        import torch

        _ = torch.__version__
    except Exception:
        # Tests that require torch will fail with explicit errors later.
        return


_warmup_torch_import()

@pytest.fixture(autouse=True)
def reset_cache(tmp_path):
    """Reset cache before each test."""
    if os.environ.get("TRADING_SKIP_TEST_CACHE_CLEAR", "0") == "1":
        yield
        return

    import data.cache as cache_module
    from config.settings import CONFIG

    old_manual = os.environ.get("TRADING_MANUAL_CACHE_DELETE")
    old_cache_dir = CONFIG._cache_dir_cached
    old_cache = cache_module._cache

    test_cache_dir = tmp_path / "cache"
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    CONFIG._cache_dir_cached = test_cache_dir
    cache_module._cache = None

    os.environ["TRADING_MANUAL_CACHE_DELETE"] = "1"
    cache = cache_module.get_cache()
    cache.clear()
    yield
    cache.clear()

    cache_module._cache = old_cache
    CONFIG._cache_dir_cached = old_cache_dir

    if old_manual is None:
        os.environ.pop("TRADING_MANUAL_CACHE_DELETE", None)
    else:
        os.environ["TRADING_MANUAL_CACHE_DELETE"] = old_manual

@pytest.fixture
def temp_model_dir(tmp_path):
    """Temporary directory for model saves (cache-safe)."""
    from config.settings import CONFIG
    CONFIG.set_model_dir(str(tmp_path))
    yield tmp_path
    CONFIG.set_model_dir("")  

@pytest.fixture(autouse=True, scope="session")
def force_offline_for_tests():
    """Make pytest deterministic and fast:
    - Avoid network calls (AkShare/Yahoo)
    - Allow tests to run even without data sources.
    """
    old = os.environ.get("TRADING_OFFLINE")
    os.environ["TRADING_OFFLINE"] = "1"
    yield
    if old is None:
        os.environ.pop("TRADING_OFFLINE", None)
    else:
        os.environ["TRADING_OFFLINE"] = old
