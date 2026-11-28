"""A real Graphiti behind GraphitiBackend with no FalkorDB and no Responses-API preflight.

The one place the tests patch how GraphitiBackend builds its driver.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver

from fused_memory.backends import graphiti_client


def _spec_driver(**_kwargs: Any) -> MagicMock:
    driver = MagicMock(spec=FalkorDriver)
    driver.clone.return_value = MagicMock(spec=FalkorDriver)
    return driver


def run_graphiti_offline(monkeypatch: pytest.MonkeyPatch, *, add_episode: Callable[..., Any]) -> None:
    """Back every GraphitiBackend with a spec'd driver double, skip its preflight, and stand *add_episode* in for Graphiti's."""
    monkeypatch.setattr(graphiti_client, '_MultiTenantFalkorDriver', _spec_driver)
    monkeypatch.setattr(graphiti_client, 'check_openai_responses_api', lambda: None)
    monkeypatch.setattr(Graphiti, 'add_episode', add_episode)
