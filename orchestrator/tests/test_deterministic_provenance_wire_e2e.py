"""Cross-package e2e: a DeterministicRunner-shaped close survives the real wire
(task 5241, review round 1).

fused-memory refuses `deterministic-*` done provenance from any caller outside
``reconciliation.deterministic_provenance_allowed_agent_prefixes`` (default
``['orchestrator']``), and refuses an unidentified caller outright. The unit
tests on either side each hard-code their half of the identity, so neither
can show that the orchestrator's real ``Scheduler.set_task_status`` actually
ARRIVES identified. This file drives that path end to end: a REAL
``Scheduler`` in front of a REAL ``TaskInterceptor`` (DEFAULT config) over a
REAL ``SqliteTaskBackend``, with a REAL ``WriteJournal`` attached.

The rig is ``test_reopen_sticks_e2e.py``'s, with ONE deliberate difference:
that file's bridge synthesizes an ``agent_id``, while this one forwards the
wire ``agent_id`` VERBATIM (see ``_build_rig``). Helpers are copied locally
rather than imported, as that file does, to avoid a cross-test-module import.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

pytest.importorskip('fused_memory')

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.services.write_journal import WriteJournal
from shared.task_metadata import DoneProvenance

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

pytestmark = pytest.mark.asyncio

_GATE_ESCALATION_ID = 'esc-5241-wire'


def _gate_provenance() -> dict:
    """The blob DeterministicRunner's pure-gate arm sends, built from the
    public shared model rather than the runner's private builder."""
    return DoneProvenance(
        kind='deterministic-gate',
        note='pure gate resolved',
        escalation_id=_GATE_ESCALATION_ID,
    ).model_dump(exclude_none=True)


def _init_git_repo(path: Path) -> None:
    """A minimal git repo at *path* with one commit (``git init`` creates it)."""
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.email', 't@e.example'], check=True)
    subprocess.run(['git', '-C', str(path), 'config', 'user.name', 'T'], check=True)
    (path / 'seed.txt').write_text('seed\n')
    subprocess.run(['git', '-C', str(path), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(path), 'commit', '-q', '-m', 'seed'], check=True)


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@dataclass
class _Rig:
    project_root: str
    backend: SqliteTaskBackend
    scheduler: Scheduler
    journal: WriteJournal

    async def add_task(self) -> str:
        added = await self.backend.add_task(project_root=self.project_root, title='T')
        return str(added['id'])

    async def read_fresh(self, task_id: str) -> dict:
        """Read *task_id* over an INDEPENDENT backend connection, so an
        assertion sees what persisted rather than ``backend``'s own state."""
        fresh = SqliteTaskBackend(TaskmasterConfig(project_root=self.project_root))
        await fresh.start()
        try:
            return await fresh.get_task(task_id, project_root=self.project_root)
        finally:
            await fresh.close()


@pytest_asyncio.fixture
async def rig(tmp_path: Path, event_buffer: EventBuffer) -> AsyncIterator[_Rig]:
    repo = tmp_path / 'repo'
    _init_git_repo(repo)
    orch_config = OrchestratorConfig(project_root=repo)
    project_root = str(orch_config.project_root)

    backend = SqliteTaskBackend(TaskmasterConfig(project_root=project_root))
    await backend.start()
    interceptor = TaskInterceptor(backend, None, event_buffer, config=FusedMemoryConfig())
    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    interceptor.set_write_journal(journal)

    async def _bridge_dispatch_tool(name: str, arguments: dict, *, timeout: float = 15) -> dict:
        """Stand-in MCP transport that forwards ``agent_id`` VERBATIM.

        This models production rather than simplifying it. fused-memory
        serves stateless HTTP, where a tools/call session never received an
        ``InitializeRequest``, so ``server/tools.py::_resolve_identity`` has
        no clientInfo to fall back to and the interceptor receives exactly
        the wire ``agent_id``, ``None`` included. The fused-memory half of
        that claim is pinned against the real SDK session by
        ``fused-memory/tests/test_task_write_agent_id.py``.
        """
        assert name == 'set_task_status', f'unexpected dispatch_tool call: {name!r}'
        payload = await interceptor.set_task_status(
            arguments['id'],
            arguments['status'],
            arguments['project_root'],
            done_provenance=arguments.get('done_provenance'),
            reopen_reason=arguments.get('reopen_reason'),
            agent_id=arguments.get('agent_id'),
        )
        return {'result': {'structuredContent': payload}}

    scheduler = Scheduler(orch_config)
    scheduler.dispatch_tool = _bridge_dispatch_tool  # type: ignore[method-assign]

    try:
        yield _Rig(
            project_root=project_root, backend=backend, scheduler=scheduler, journal=journal,
        )
    finally:
        await journal.close()
        await backend.close()


class TestDeterministicProvenanceOverTheWire:
    async def test_the_scheduler_close_is_accepted_and_journals_the_orchestrator(
        self, rig: _Rig,
    ) -> None:
        task_id = await rig.add_task()

        await rig.scheduler.set_task_status(task_id, 'done', done_provenance=_gate_provenance())

        persisted = await rig.read_fresh(task_id)
        assert persisted['status'] == 'done'
        provenance = persisted['metadata']['done_provenance']
        assert provenance['kind'] == 'deterministic-gate'
        assert provenance['escalation_id'] == _GATE_ESCALATION_ID

        rows = [
            row for row in await rig.journal.get_ops_since('1970-01-01')
            if row['operation'] == 'set_task_status'
        ]
        assert len(rows) == 1, rows
        assert rows[0]['agent_id'] == 'orchestrator'
        assert json.loads(rows[0]['params'])['done_provenance_kind'] == 'deterministic-gate'

    async def test_the_same_close_without_an_identity_is_refused(self, rig: _Rig) -> None:
        """Negative control: the acceptance above is caused by the identity
        the Scheduler sends, not by a permissive rig."""
        task_id = await rig.add_task()

        response = await rig.scheduler.dispatch_tool('set_task_status', {
            'id': task_id,
            'status': 'done',
            'project_root': rig.project_root,
            'done_provenance': _gate_provenance(),
        })

        refusal = response['result']['structuredContent']
        assert refusal.get('error_type') == 'DeterministicProvenanceCallerNotPermitted', refusal
        assert (await rig.read_fresh(task_id))['status'] != 'done'
