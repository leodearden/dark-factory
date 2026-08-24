"""Audit of Stage 2's self-reported cycle stats (task 3051).

**The standard.** A cycle stat may only increment after the underlying MCP
operation is confirmed to have succeeded — never on LLM self-report alone.
This module records which Stage 2 counters already meet that standard, which
do not, and pins the corroboration layer that closes the gap for the
task-filing path.

**Journal-derived counters — already confirmed-action-only, NOT in scope.**
``memories_added``, ``memories_deleted``, ``episodes_added``,
``episodes_deleted``, ``edges_updated``, ``entity_summaries_refreshed``,
``entities_merged``, ``entity_summaries_rebuilt``, ``dead_letters_replayed``,
``episodes_replayed`` and ``graphiti_writes_queued`` are recomputed from the
write journal by ``derive_stage_stats`` (``stage_stats._OP_TO_STAT`` /
``_COMPUTED_STAT_KEYS``) and rewritten by ``stats_verifier``, so an LLM
self-report cannot inflate any of them. ``edges_updated`` additionally gates
on a server-side readback (``stage_stats._count_update_edge``) — precisely
the confirmed-action-only standard, already met.

**Purely self-reported counters — no journal ground truth.**
``submit_task``/``resolve_ticket`` are not journaled
(``TaskInterceptor._journal_around`` wraps only ``set_task_status`` /
``update_task`` / ``remove_tasks`` / ``add_dependency`` /
``remove_dependency``), so nothing recomputes these:

1. ``tasks_created`` — repaired UPWARD from ``task_created_records`` by
   ``TaskKnowledgeSync._apply_post_flight_guards`` (task 3046). That record
   list was itself pure LLM self-report and was never corroborated, so a
   hallucinated record could inflate the counter with no external check.
   **This is the defect task 3051 closes**: each record's
   ``(project_id, task_id)`` key is now resolved to that project's own root
   via ``known_projects`` and confirmed with ``taskmaster.get_task`` +
   ``flag_dedup.confirm_task_present`` before it may raise the counter.
2. ``tasks_hints_updated`` — the only other purely self-reported counter in
   the proactive / cross-project filing path. ``prompts/stage2.py`` already
   mandates a confirmed-action-only rule for it (re-read via ``get_task``,
   increment only when the returned ``memory_hints`` is a SUPERSET), so it
   meets the standard at the prompt layer; it lacks an action-record list and
   framework corroboration, which is a prompt-contract expansion filed as a
   follow-up rather than absorbed here.
3. ``stage1_mem0_flags_processed`` / ``stage1_analytical_findings_processed``
   — flag-processing counters outside the task-filing path. The clamps that
   once checked them were removed by tasks 2229/2230 (W5-mu);
   ``_apply_post_flight_guards`` now only ``setdefault(..., 0)``s them. That
   audit conclusion is pinned as executable behaviour below so the prompt's
   description of the framework cannot silently drift back out of sync.
4. ``memories_written`` — a retired counter (``stats_verifier``) the prompt
   still asks for. Not load-bearing, no repair path, deliberately untouched.

**Fail direction.** Corroboration is fail-CLOSED on the increment (an
unverifiable record does not count) while remaining fail-SAFE for the stage:
because the repair is upward-only, dropping an uncorroborated record merely
leaves the self-reported value alone — it never lowers a counter, never
raises, never aborts the stage. A configuration that cannot corroborate at
all is made loud rather than silent (WARNING + explicit
``task_created_records_uncorroborated`` / ``_unresolvable`` stats).

Per the ``test_recon_gate_closure_guidance.py`` house rule, prompt prose is
intentionally NOT pinned here — only machine-readable contract tokens and
code behaviour.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync

# ── Shared harness ───────────────────────────────────────────────────────────
#
# Replicated from tests/test_stage2_tasks_created_accounting.py:321-359 rather
# than cross-imported, matching the precedent that module's own docstring
# records (and test_recon_gate_closure_guidance.py::_make_consolidator before
# it): a module-private fixture is not a shared contract. Extended here with a
# configurable ``known_projects`` mapping and a taskmaster stub that answers
# ``get_task`` per ``(task_id, project_root)``, which the corroboration pass
# needs.

_TEST_PROJECT_ID = 'test_project'
_TEST_PROJECT_ROOT = '/tmp/test'

_REPAIR_LOGGER = 'fused_memory.reconciliation.stages.task_knowledge_sync'


def _task_record(task_id: object, title: str = 'A real task', status: str = 'pending') -> dict:
    """A get_task result that ``confirm_task_present`` classifies as PRESENT."""
    return {'id': str(task_id), 'title': title, 'status': status}


def _not_found(task_id: object = '0') -> dict:
    """The canonical not-found error dict a missing task resolves to."""
    return {
        'error': f'No tasks found for ID(s): {task_id}',
        'error_type': 'TaskmasterError',
    }


class _StubTaskmaster:
    """Taskmaster double whose ``get_task`` answers per ``(task_id, root)``.

    *results* is keyed on the exact ``(str(task_id), project_root)`` pair the
    lookup is expected to issue, so a lookup routed to the WRONG project root
    cannot accidentally satisfy an assertion. Unmapped keys fall back to
    *default* (a not-found error dict). A mapped value that is an ``Exception``
    instance is RAISED, which drives the raised-exception path.

    Every call is appended to ``calls``; ``max_concurrent`` records the peak
    number of simultaneously in-flight lookups, so a test can tell a single
    concurrent batch from a sequential await loop.
    """

    def __init__(
        self,
        results: dict[tuple[str, str], Any] | None = None,
        default: Any = None,
    ) -> None:
        self.results: dict[tuple[str, str], Any] = dict(results or {})
        self.default: Any = _not_found() if default is None else default
        self.calls: list[tuple[Any, str]] = []
        self.max_concurrent = 0
        self._in_flight = 0

    async def get_task(self, task_id: Any, project_root: str) -> Any:
        self.calls.append((task_id, project_root))
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            # Yield once so concurrently-gathered lookups genuinely overlap.
            await asyncio.sleep(0)
            result = self.results.get((str(task_id), project_root), self.default)
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self._in_flight -= 1


def _mock_deps(
    known_projects: dict[str, str] | None = None,
    taskmaster: Any = None,
) -> dict:
    """Kwargs for constructing a TaskKnowledgeSync with mocked deps."""
    config = ReconciliationConfig(enabled=True, explore_codebase_root=_TEST_PROJECT_ROOT)
    write_journal_mock = MagicMock()
    write_journal_mock.get_ops_by_causation = AsyncMock(return_value=[])
    journal_mock = MagicMock()
    journal_mock.write_journal = write_journal_mock
    journal_mock.record_run_session = AsyncMock()
    journal_mock.clear_run_session = AsyncMock()
    return {
        'memory_service': AsyncMock(),
        'taskmaster': AsyncMock() if taskmaster is None else taskmaster,
        'journal': journal_mock,
        'config': config,
        'scope': ProjectScope(ProjectId(_TEST_PROJECT_ID), ProjectRoot(_TEST_PROJECT_ROOT)),
        'known_projects': dict(known_projects) if known_projects else {},
    }


def _make_stage(
    known_projects: dict[str, str] | None = None,
    taskmaster: Any = None,
) -> TaskKnowledgeSync:
    return TaskKnowledgeSync(
        StageId.task_knowledge_sync,
        **_mock_deps(known_projects=known_projects, taskmaster=taskmaster),
    )


def _make_report(stats: dict) -> StageReport:
    now = datetime.now(tz=UTC)
    return StageReport(
        stage=StageId.task_knowledge_sync,
        started_at=now,
        completed_at=now,
        stats=stats,
    )


class TestAuditHarness:
    """Self-check on the shared harness above (prerequisite pre-1)."""

    def test_make_stage_wires_known_projects_and_taskmaster(self):
        stub = _StubTaskmaster()
        stage = _make_stage(known_projects={'dark_factory': '/repo/df'}, taskmaster=stub)

        assert stage.known_projects == {'dark_factory': '/repo/df'}
        assert stage.taskmaster is stub
        assert stage.project_id == _TEST_PROJECT_ID

    def test_defaults_leave_known_projects_empty(self):
        stage = _make_stage()

        assert stage.known_projects == {}

    @pytest.mark.asyncio
    async def test_stub_answers_per_task_id_and_root(self):
        """A configured (task_id, root) pair resolves; any other pair — same id
        under a DIFFERENT root included — falls back to not-found."""
        stub = _StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')})

        hit = await stub.get_task('3045', '/repo/df')
        miss_root = await stub.get_task('3045', '/repo/other')
        miss_id = await stub.get_task('9999', '/repo/df')

        assert hit == _task_record('3045')
        assert miss_root['error_type'] == 'TaskmasterError'
        assert miss_id['error_type'] == 'TaskmasterError'
        assert stub.calls == [
            ('3045', '/repo/df'),
            ('3045', '/repo/other'),
            ('9999', '/repo/df'),
        ]

    @pytest.mark.asyncio
    async def test_stub_raises_a_mapped_exception(self):
        boom = RuntimeError('backend down')
        stub = _StubTaskmaster(results={('3045', '/repo/df'): boom})

        with pytest.raises(RuntimeError, match='backend down'):
            await stub.get_task('3045', '/repo/df')

    @pytest.mark.asyncio
    async def test_stub_tracks_peak_concurrency(self):
        """``max_concurrent`` distinguishes one gathered batch (== N) from a
        sequential await loop (== 1), which the corroboration pins rely on."""
        stub = _StubTaskmaster()

        await asyncio.gather(*(stub.get_task(i, '/repo/df') for i in range(4)))

        assert stub.max_concurrent == 4
