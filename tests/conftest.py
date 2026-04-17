import pytest
from edel.config.defaults import RUN_CONFIG

@pytest.fixture
def base_run_config():
    """Return a fresh copy of the default run configuration."""
    return RUN_CONFIG.copy()
