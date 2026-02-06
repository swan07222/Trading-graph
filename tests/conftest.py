# tests/conftest.py
"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix config import for all tests
import config.settings as config_module
sys.modules['config'] = config_module


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