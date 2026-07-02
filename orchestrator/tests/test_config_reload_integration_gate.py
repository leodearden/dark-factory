"""δ integration gate: config hot-reload boundary scenarios (task 2008).

Task alpha (config.py RELOADABLE_FIELDS / diff_config / apply_reload), beta
(harness.py Harness.reload_config), and gamma (escalation/server.py
reload_config MCP tool) are fully merged on main — see
plans/config-hot-reload-prd.md. This module is a green-on-arrival
integration gate, not a RED→GREEN pair: it drives the REAL load_config()
off a real on-disk orchestrator.yaml (never mocking load_config, unlike
test_harness_config_reload.py's beta tests) and observes reload semantics
through downstream consumers beta's mocked-load tests never touch — the
next agent spawn's model, the dispatch-semaphore sizing, and the
offline-lane worker's test-thread count + GitOps identity.

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from _workflow_helpers import FakeBriefing, FakeScheduler

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.config import load_config
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# ---------------------------------------------------------------------------
# Rig — pinned starting values for the on-disk orchestrator.yaml. Chosen to
# be unambiguously test-only (distinct from defaults.yaml's packaged
# values), so each scenario's "new" value is a plain, explicit literal
# rather than something derived from the repo's tracked config.
# ---------------------------------------------------------------------------

_PINNED_IMPLEMENTER = 'sonnet'
_PINNED_MAX_CONCURRENT_TASKS = 5
_PINNED_OFFLINE_LANE_TEST_THREADS = 4


def _make_reload_harness(tmp_path: Path, monkeypatch) -> tuple[Harness, Path]:
    """Build a Harness whose config is loaded for real off an on-disk YAML.

    Mirrors test_harness_config_reload.py::_make_harness_with_mocks (a real
    Scheduler/GitOps via Harness(config), a MagicMock RunStore, a real
    EventStore) but — unlike that helper, which builds harness.config
    directly via ``OrchestratorConfig(project_root=tmp_path)`` against
    whatever ``config.yaml`` the test process's cwd happens to resolve —
    constructs harness.config via the REAL ``load_config()`` reading a
    project-local ``orchestrator.yaml`` this helper writes at *tmp_path*.
    Live provenance therefore equals disk, so a later ``_edit_yaml`` +
    ``harness.reload_config()`` call produces a clean, single-leaf diff
    instead of depending on the repo's tracked config.yaml.

    ``ORCH_CONFIG_PATH`` is set via ``monkeypatch.setenv`` (re-establishing
    it after conftest's autouse ``_isolate_orch_config`` deletes it) and
    stays set for the rest of the test, so a later ``reload_config()`` call
    re-reads the SAME file.
    """
    config_path = tmp_path / 'orchestrator.yaml'
    config_path.write_text(yaml.safe_dump({
        'models': {'implementer': _PINNED_IMPLEMENTER},
        'max_concurrent_tasks': _PINNED_MAX_CONCURRENT_TASKS,
        'git': {'offline_lane_test_threads': _PINNED_OFFLINE_LANE_TEST_THREADS},
    }))
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))

    config = load_config()
    harness = Harness(config)

    # Inject a mock RunStore so persistence calls never touch a real DB, and
    # a real EventStore pointing at a tmp DB so config_reload rows are
    # queryable — same injection convention as _make_harness_with_mocks.
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


def _query_config_reload_events(event_store: EventStore) -> list[dict]:
    """Query all config_reload event rows from *event_store*'s DB.

    Mirrors test_harness_config_reload.py::_query_events, pinned to the
    'config_reload' event type this gate exercises exclusively.
    """
    conn = sqlite3.connect(str(event_store.db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM events WHERE event_type = 'config_reload' ORDER BY id",
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _other_model(current: str) -> str:
    """Return a model name distinct from *current* (mirrors test_harness_config_reload.py:74)."""
    return 'opus' if current != 'opus' else 'sonnet'


class StubInvoke:
    """Fake invoke_agent capturing per-call kwargs (model, in particular).

    Mirrors test_workflow_e2e.py:260 AgentStub.invoke_agent's signature so
    it can be patched onto orchestrator.workflow.invoke_agent, but only
    records the model kwarg per call rather than performing file/git side
    effects — these scenarios drive TaskWorkflow._invoke(...) directly (not
    the full PLAN/EXECUTE/... state machine), so no worktree is needed.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def invoke_agent(
        self,
        prompt: str,
        system_prompt: str,
        cwd: Path,
        model: str = 'opus',
        max_turns: int = 50,
        max_budget_usd: float = 5.0,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_config: dict | None = None,
        output_schema: dict | None = None,
        permission_mode: str = 'bypassPermissions',
        sandbox_modules: list[str] | None = None,
        effort: str | None = None,
        backend: str = 'claude',
        oauth_token: str | None = None,
        resume_session_id: str | None = None,
        session_id: str | None = None,
        timeout_seconds: float | None = None,
        config_dir: Path | None = None,
        env_overrides: dict[str, str] | None = None,
        startup_grace_secs: float = 120.0,
    ) -> AgentResult:
        self.calls.append({'model': model})
        return AgentResult(success=True, output='OK')


def _make_reload_workflow(
    harness: Harness, tmp_path: Path, monkeypatch,
) -> tuple[TaskWorkflow, StubInvoke]:
    """Build a TaskWorkflow sharing harness.config BY REFERENCE, invoke_agent stubbed.

    Sharing the config object (rather than a copy) means a leaf mutated in
    place by harness.reload_config() (I3 identity preservation) is observed
    by workflow._invoke on its NEXT call with no extra wiring — there is no
    separate config to go stale.
    """
    stub = StubInvoke()
    monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
    assignment = TaskAssignment(task_id='42', task={'id': '42'}, modules=[])
    workflow = TaskWorkflow(
        assignment=assignment,
        config=harness.config,
        git_ops=GitOps(harness.config.git, tmp_path),
        scheduler=FakeScheduler(),
        briefing=FakeBriefing(),
        mcp=None,
    )
    return workflow, stub


# ---------------------------------------------------------------------------
# Scenario 1 (I1, I7): a real load_config() failure off a corrupted on-disk
# YAML — the surface test_harness_config_reload.py's mocked-load-raises test
# never exercises (it patches orchestrator.harness.load_config directly).
# ---------------------------------------------------------------------------


class TestS1BadYamlFailClosed:
    """PRD boundary scenario 1: a genuinely corrupt on-disk config fails closed."""

    @pytest.mark.asyncio
    async def test_corrupt_yaml_fails_closed(
        self, tmp_path: Path, monkeypatch, caplog,
    ) -> None:
        harness, config_path = _make_reload_harness(tmp_path, monkeypatch)
        before = harness.config.model_dump_json()

        # Genuinely invalid YAML syntax (unterminated flow sequence) — a real
        # yaml.safe_load() parse error surfacing through load_config(), not a
        # mocked exception.
        config_path.write_text('models: [unterminated')

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            report = await harness.reload_config()

        assert report['reloaded'] is False
        assert report['config_path'] == str(config_path)
        assert report['applied'] == {}
        assert report['restart_required'] == {}
        assert report['unchanged'] == 0
        assert report['error']

        assert harness.config.model_dump_json() == before, (
            'live config must be byte-identical after a failed real-disk load (I1)'
        )

        rows = _query_config_reload_events(harness.event_store)
        assert len(rows) == 1, f'Expected exactly one config_reload event row; got {rows!r}'
        assert json.loads(rows[0]['data']) == report

        harness_warnings = [
            r for r in caplog.records
            if r.name == 'orchestrator.harness' and r.levelno >= logging.WARNING
        ]
        assert harness_warnings, 'Expected at least one WARNING on a failed reload'


# ---------------------------------------------------------------------------
# Scenario 2 (read-at-spawn, workflow.py:6801): an allowlisted models.*
# edit must be observed by the NEXT agent spawn — a surface
# test_harness_config_reload.py never touches, since its tests only assert
# on harness.config and never drive a TaskWorkflow invocation against the
# reloaded config.
# ---------------------------------------------------------------------------


class TestS2AllowlistedNextSpawn:
    """PRD boundary scenario 2: an allowlisted models.implementer edit reaches the next spawn."""

    @pytest.mark.asyncio
    async def test_allowlisted_model_edit_applies_to_next_spawn(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        harness, config_path = _make_reload_harness(tmp_path, monkeypatch)
        workflow, stub = _make_reload_workflow(harness, tmp_path, monkeypatch)

        live_implementer = harness.config.models.implementer
        new_implementer = _other_model(live_implementer)
        _edit_yaml(config_path, 'models.implementer', new_implementer)

        report = await harness.reload_config()

        assert report['reloaded'] is True
        assert report['applied']['models.implementer'] == {
            'old': live_implementer, 'new': new_implementer,
        }
        assert harness.config.models.implementer == new_implementer

        result = await workflow._invoke(IMPLEMENTER, 'do the task', tmp_path)

        assert result.success is True
        assert stub.calls, 'Expected the stubbed invoke_agent to have been called'
        assert stub.calls[-1]['model'] == new_implementer, (
            'the next spawn must read models.implementer fresh at call time, '
            'observing the reloaded value rather than one captured earlier'
        )
