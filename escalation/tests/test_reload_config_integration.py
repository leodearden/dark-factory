"""Escalation-side integration gate for config hot-reload (task 2008, δ).

Task gamma (escalation/server.py reload_config MCP tool, task 2007) already
covers the standalone-guard and a SimpleNamespace-stubbed wired path — see
test_server.py::TestReloadConfigTool. This module drives the reload through
the FULL stack — tool -> a REAL Harness -> apply_reload -> a real on-disk
orchestrator.yaml — the "observed through the tool response" user-observable
surface gamma's stubbed harness never exercises.

Scenario 7 (standalone harness=None -> error dict) is intentionally NOT
retested here: already green in gamma (test_server.py:3804,
test_standalone_returns_wired_error).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports.
# Mirrors test_server.py lines 30-54 / test_merge_status_git_authority.py
# lines 33-45. Guarded so the rest of the file (none, currently — this
# whole module is orchestrator-dependent) still collects when the
# orchestrator package is absent (e.g. an escalation-only install).
# ---------------------------------------------------------------------------
try:
    from orchestrator.config import load_config  # type: ignore[reportMissingImports]
    from orchestrator.event_store import EventStore  # type: ignore[reportMissingImports]
    from orchestrator.harness import Harness  # type: ignore[reportMissingImports]
    from orchestrator.run_store import RunStore  # type: ignore[reportMissingImports]
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    # Satisfy pyright's definite-assignment check — the class below is
    # guarded by @pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE) so these
    # stubs are never exercised at runtime.
    load_config: Any = None
    EventStore: Any = None
    Harness: Any = None
    RunStore: Any = None


async def _call_tool(server: Any, name: str, **kwargs: Any) -> Any:
    """Invoke an async MCP tool by name (mirrors test_server.py:3661)."""
    tool = await server.get_tool(name)
    return await tool.fn(**kwargs)


def _make_reload_harness(tmp_path: Path, monkeypatch) -> tuple[Any, Path]:
    """Build a real Harness whose config loads off an on-disk orchestrator.yaml.

    Replicated (not imported) from
    orchestrator/tests/test_config_reload_integration_gate.py's helper of
    the same name — escalation/tests has no sys.path access to
    orchestrator/tests, and harness-reload test helpers are replicated
    rather than cross-imported by established convention (see that
    module's docstring).

    Unlike orchestrator/tests/conftest.py's autouse ``_isolate_orch_config``
    fixture (which pins ``ORCH_PROJECT_ROOT`` and deletes
    ``ORCH_CONFIG_PATH`` for every orchestrator test), escalation/tests has
    no such fixture, so both env vars are set explicitly here.
    """
    config_path = tmp_path / 'orchestrator.yaml'
    config_path.write_text(yaml.safe_dump({
        'models': {'implementer': 'sonnet'},
        'max_concurrent_tasks': 5,
    }))
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))
    monkeypatch.setenv('ORCH_PROJECT_ROOT', str(tmp_path))

    config = load_config()
    harness = Harness(config)
    harness._run_store = MagicMock(spec=RunStore)
    harness._run_id = 'run-test-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-test-0001')
    return harness, config_path


def _edit_yaml(path: Path, dotted_leaf: str, value: Any) -> None:
    """Rewrite one dotted leaf on the YAML file at *path*, preserving the rest."""
    data = yaml.safe_load(path.read_text()) or {}
    node = data
    parts = dotted_leaf.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    path.write_text(yaml.safe_dump(data))


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestReloadConfigToolIntegration:
    """reload_config MCP tool driven end-to-end through a REAL harness (task 2008 δ).

    gamma's TestReloadConfigTool (test_server.py:3788) proves the tool is a
    thin standalone-guard + verbatim delegate using a SimpleNamespace stub
    harness. This proves the delegate target is a REAL Harness whose
    reload_config() drives the real apply_reload/load_config stack end to
    end — the surface actually observable through the tool response.
    """

    async def test_tool_returns_verbatim_report_from_real_harness(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        harness, config_path = _make_reload_harness(tmp_path, monkeypatch)
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, harness=harness)

        live_implementer = harness.config.models.implementer
        new_implementer = 'opus' if live_implementer != 'opus' else 'sonnet'
        live_mct = harness.config.max_concurrent_tasks

        # One allowlisted (models.implementer) + one non-allowlisted
        # (max_concurrent_tasks) edit in the same reload — exercises both
        # dispositions through the tool response in one call.
        _edit_yaml(config_path, 'models.implementer', new_implementer)
        _edit_yaml(config_path, 'max_concurrent_tasks', live_mct + 1)

        result = await _call_tool(server, 'reload_config')

        assert result['reloaded'] is True
        assert result['config_path'] == str(config_path)
        assert result['applied'] == {
            'models.implementer': {'old': live_implementer, 'new': new_implementer},
        }
        assert result['restart_required'] == {
            'max_concurrent_tasks': {'old': live_mct, 'new': live_mct + 1},
        }
        assert result['error'] is None

        assert harness.config.models.implementer == new_implementer, (
            'the allowlisted leaf must be mutated on the real harness config'
        )
        assert harness.config.max_concurrent_tasks == live_mct, (
            'the non-allowlisted leaf must NOT be mutated on the real harness config (I2)'
        )
