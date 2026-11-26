"""Subject: conftest.py::_isolate_mcp_lifecycle_session (task 9528)."""

from unittest.mock import MagicMock

import pytest

import orchestrator.mcp_lifecycle
from orchestrator.mcp_lifecycle import get_session


@pytest.fixture(scope='module')
def session_left_by_an_earlier_test():
    """Module scope sets up before every function fixture, like an earlier test's McpLifecycle.start() under a patched McpSession."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator.mcp_lifecycle, '_session', MagicMock(name='leaked McpSession'))
        yield


@pytest.mark.usefixtures('session_left_by_an_earlier_test')
def test_a_session_an_earlier_test_left_is_not_visible():
    with pytest.raises(RuntimeError):
        get_session()
