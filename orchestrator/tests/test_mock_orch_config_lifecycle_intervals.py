"""mock_orch_config gives every registered lifecycle sweep a real interval.

A ``BackgroundService`` sleeps ``interval_secs`` between passes, so a
MagicMock interval makes every pass fail with TypeError and park in backoff.
This fails loudly when a new service reads a config field the fixture leaves
unset.
"""
from __future__ import annotations

from unittest.mock import patch

from orchestrator.background_service import BackgroundService
from orchestrator.harness import Harness


def test_every_registered_sweep_has_a_real_interval(mock_orch_config):
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        harness = Harness(mock_orch_config)
    harness._build_lifecycle_registry()

    assert harness._lifecycle is not None
    sweeps = [s for s in harness._lifecycle.services if isinstance(s, BackgroundService)]
    unset = {s.name: s.interval_secs for s in sweeps if type(s.interval_secs) not in (int, float)}
    assert sweeps
    assert unset == {}
