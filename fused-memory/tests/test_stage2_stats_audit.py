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
   That proves the task EXISTS in that project, not that this cycle created
   it, so it closes the fabricated-id hole only; binding a record to this
   run's own creation is task 4873's scope.
2. ``tasks_hints_updated`` — the only other purely self-reported counter in
   the proactive / cross-project filing path. ``prompts/stage2.py`` already
   mandates a confirmed-action-only rule for it (re-read via ``get_task``,
   increment only when the returned ``memory_hints`` is a SUPERSET), so it
   meets the standard at the prompt layer; it lacks an action-record list and
   framework corroboration, which is a prompt-contract expansion filed as
   task 4018 (now in task 4873's scope) rather than absorbed here.
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
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    _TASK_CREATED_SUCCESS_STATUSES,
    TaskKnowledgeSync,
    _action_record_keys,
    _corroborate_record_keys,
)

# ── Shared harness ───────────────────────────────────────────────────────────
#
# Replicated from tests/test_stage2_tasks_created_accounting.py::_AllPresentTaskmaster
# and ::_mock_deps rather
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


# ── step-3: _action_record_keys ──────────────────────────────────────────────


class _ExplodingRecord(dict):
    """A dict-shaped record that raises while being inspected."""

    def get(self, key, default=None):  # type: ignore[override]
        raise ValueError(f'record inspection exploded on {key!r}')


class TestActionRecordKeys:
    """``_action_record_keys(records, default_project_id, valid_statuses)``.

    The corroboration pass needs the actual ``(project_id, task_id)`` pairs, not
    just how many there are, so task 3046's counting logic is factored into a
    helper that RETURNS the deduped key set. Every rule 3046 established must
    survive that move — these pins are the guard against silent drift, and
    ``_count_valid_task_created_records`` becomes ``len()`` of this.

    ``valid_statuses`` is a parameter (not a hardcoded constant read) because
    task 4018's ``tasks_hints_updated`` records work reuses this helper with
    its own accepted-status vocabulary.
    """

    def test_returns_a_set_of_project_task_pairs(self):
        keys = _action_record_keys(
            [
                {'action': 'task_created', 'task_id': '3045', 'status': 'created',
                 'project_id': 'dark_factory'},
            ],
        )

        assert keys == {('dark_factory', '3045')}
        assert isinstance(keys, set)

    @pytest.mark.parametrize(
        'status',
        ['created', 'CREATED', ' Created ', 'combined', 'COMBINED', '\tcombined\n'],
    )
    def test_status_match_is_case_and_whitespace_insensitive(self, status):
        keys = _action_record_keys(
            [{'task_id': '1', 'status': status, 'project_id': 'p'}],
        )

        assert keys == {('p', '1')}

    @pytest.mark.parametrize('status', ['failed', 'FAILED', ' failed ', 'pending', '', None, 7])
    def test_non_accepted_status_never_keys_even_with_a_task_id(self, status):
        """`failed` is never counted regardless of a present task_id."""
        keys = _action_record_keys(
            [{'task_id': '3045', 'status': status, 'project_id': 'p'}],
        )

        assert keys == set()

    @pytest.mark.parametrize('task_id', [None, '', '   '])
    def test_missing_or_blank_task_id_is_skipped(self, task_id):
        keys = _action_record_keys(
            [{'task_id': task_id, 'status': 'created', 'project_id': 'p'}],
        )

        assert keys == set()

    def test_same_numeric_id_under_two_projects_is_two_keys(self):
        """Taskmaster ids are per-project, so cross-project routing filing id
        3045 into two projects is TWO tasks, not one duplicate report."""
        keys = _action_record_keys(
            [
                {'task_id': 3045, 'status': 'created', 'project_id': 'dark_factory'},
                {'task_id': '3045', 'status': 'created', 'project_id': 'reify'},
            ],
        )

        assert keys == {('dark_factory', '3045'), ('reify', '3045')}

    def test_duplicate_records_collapse_to_one_key(self):
        keys = _action_record_keys(
            [
                {'task_id': 3045, 'status': 'created', 'project_id': 'dark_factory'},
                {'task_id': ' 3045 ', 'status': 'combined', 'project_id': ' dark_factory'},
            ],
        )

        assert keys == {('dark_factory', '3045')}

    @pytest.mark.parametrize('project_id', [None, '', '   '])
    def test_missing_or_blank_project_id_falls_back_to_default(self, project_id):
        """An omitted project_id must not masquerade as a second, distinct
        cross-project filing of the same task."""
        keys = _action_record_keys(
            [
                {'task_id': '3045', 'status': 'created', 'project_id': project_id},
                {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            ],
            default_project_id='dark_factory',
        )

        assert keys == {('dark_factory', '3045')}

    def test_absent_project_id_key_falls_back_to_default(self):
        keys = _action_record_keys(
            [{'task_id': '3045', 'status': 'created'}],
            default_project_id='dark_factory',
        )

        assert keys == {('dark_factory', '3045')}

    def test_default_project_id_defaults_to_none(self):
        keys = _action_record_keys([{'task_id': '3045', 'status': 'created'}])

        assert keys == {(None, '3045')}

    @pytest.mark.parametrize(
        'records',
        [None, 'not-a-list', 42, {}, {'task_id': '1'}, [], ()],
        ids=['none', 'str', 'int', 'empty-dict', 'dict', 'empty-list', 'tuple'],
    )
    def test_non_list_or_empty_input_returns_empty_set(self, records):
        assert _action_record_keys(records) == set()

    def test_non_dict_and_exploding_entries_are_skipped_not_propagated(self):
        """A malformed record degrades to 'not counted', never to an exception
        that would corrupt an otherwise-good stage report."""
        keys = _action_record_keys(
            [
                None,
                'x',
                ['a'],
                _ExplodingRecord(task_id='9', status='created', project_id='p'),
                {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            ],
        )

        assert keys == {('dark_factory', '3045')}

    def test_valid_statuses_parameter_is_honoured(self):
        """A custom accepted-status set changes which records key — the seam
        task 4018's tasks_hints_updated work reuses."""
        records = [
            {'task_id': '1', 'status': 'created', 'project_id': 'p'},
            {'task_id': '2', 'status': 'updated', 'project_id': 'p'},
        ]

        assert _action_record_keys(records) == {('p', '1')}
        assert _action_record_keys(records, valid_statuses=frozenset({'updated'})) == {('p', '2')}

    def test_default_valid_statuses_is_the_task_created_vocabulary(self):
        records = [
            {'task_id': str(i), 'status': status, 'project_id': 'p'}
            for i, status in enumerate(sorted(_TASK_CREATED_SUCCESS_STATUSES))
        ]

        assert _action_record_keys(records) == {
            ('p', str(i)) for i in range(len(_TASK_CREATED_SUCCESS_STATUSES))
        }


# ── step-5: _corroborate_record_keys ─────────────────────────────────────────


class _ExplodingProjects(dict):
    """A known_projects mapping that raises while being resolved."""

    def get(self, key, default=None):  # type: ignore[override]
        raise ValueError(f'project resolution exploded on {key!r}')


def _totals_account_for_every_key(result, keys) -> bool:
    """No key may be silently dropped: the three buckets must partition them."""
    return result.corroborated + result.uncorroborated + result.unresolvable == len(keys)


class TestCorroborateRecordKeys:
    """``_corroborate_record_keys(taskmaster, known_projects, keys)`` (task 3051).

    The corroboration pass that makes ``tasks_created``'s upward repair
    existence-checked (a key must name a task that exists in its own project;
    it need not have been created this cycle — see the function docstring): each ``(project_id, task_id)`` key is resolved to
    its OWN project root via ``known_projects`` and confirmed with
    ``taskmaster.get_task`` + ``flag_dedup.confirm_task_present``.

    Fail-CLOSED on the increment (an unverifiable key never counts as
    corroborated) but fail-SAFE for the stage: it never raises, so a bug here
    can only ever withhold a repair, never abort an otherwise-good report.
    """

    @pytest.mark.asyncio
    async def test_present_task_in_a_resolvable_project_is_corroborated(self):
        stub = _StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')})
        keys = {('dark_factory', '3045')}

        result = await _corroborate_record_keys(stub, {'dark_factory': '/repo/df'}, keys)

        assert result.corroborated == 1
        assert result.corroborated_keys == {('dark_factory', '3045')}
        assert result.uncorroborated == 0
        assert result.unresolvable == 0
        assert _totals_account_for_every_key(result, keys)
        assert stub.calls == [('3045', '/repo/df')]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'get_task_result',
        [
            {'error': 'No tasks found for ID(s): 3045', 'error_type': 'TaskmasterError'},
            {'error': 'Connection timeout', 'error_type': 'TimeoutError'},
            None,
            {},
            'a task exists',
            42,
            [],
        ],
        ids=['not-found', 'generic-error', 'none', 'empty-dict', 'str', 'int', 'list'],
    )
    async def test_non_present_results_are_uncorroborated_never_corroborated(
        self, get_task_result,
    ):
        """Absent, inconclusive, and non-dict results all fail closed."""
        stub = _StubTaskmaster(results={('3045', '/repo/df'): get_task_result})
        keys = {('dark_factory', '3045')}

        result = await _corroborate_record_keys(stub, {'dark_factory': '/repo/df'}, keys)

        assert result.corroborated == 0
        assert result.corroborated_keys == set()
        assert result.uncorroborated == 1
        assert result.unresolvable == 0
        assert _totals_account_for_every_key(result, keys)

    @pytest.mark.asyncio
    async def test_raising_get_task_is_uncorroborated_not_an_exception(self):
        stub = _StubTaskmaster(results={('3045', '/repo/df'): RuntimeError('backend down')})
        keys = {('dark_factory', '3045')}

        result = await _corroborate_record_keys(stub, {'dark_factory': '/repo/df'}, keys)

        assert result.corroborated == 0
        assert result.uncorroborated == 1
        assert result.unresolvable == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize('project_id', ['unknown_project', None, ''])
    async def test_unresolvable_project_issues_no_lookup(self, project_id):
        """A project we cannot resolve to a root is UNRESOLVABLE ('we could not
        check'), distinct from uncorroborated ('we checked and it is not
        there') — and costs no get_task round-trip."""
        stub = _StubTaskmaster()
        keys = {(project_id, '3045')}

        result = await _corroborate_record_keys(stub, {'dark_factory': '/repo/df'}, keys)

        assert result.corroborated == 0
        assert result.uncorroborated == 0
        assert result.unresolvable == 1
        assert _totals_account_for_every_key(result, keys)
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_each_key_is_looked_up_in_its_own_project_root(self):
        """A cross-project key is checked against ITS project, not this
        stage's — Taskmaster ids are per-project, so the wrong root would
        routinely land on an unrelated task sharing the id."""
        stub = _StubTaskmaster(results={
            ('3045', '/repo/df'): _task_record('3045'),
            ('3045', '/repo/reify'): _not_found('3045'),
        })
        keys = {('dark_factory', '3045'), ('reify', '3045')}

        result = await _corroborate_record_keys(
            stub, {'dark_factory': '/repo/df', 'reify': '/repo/reify'}, keys,
        )

        assert result.corroborated_keys == {('dark_factory', '3045')}
        assert result.corroborated == 1
        assert result.uncorroborated == 1
        assert result.unresolvable == 0
        assert sorted(stub.calls) == [('3045', '/repo/df'), ('3045', '/repo/reify')]

    @pytest.mark.asyncio
    async def test_mixed_batch_partitions_every_key(self):
        stub = _StubTaskmaster(results={
            ('1', '/repo/df'): _task_record('1'),
            ('2', '/repo/df'): _not_found('2'),
        })
        keys = {('dark_factory', '1'), ('dark_factory', '2'), ('nowhere', '3')}

        result = await _corroborate_record_keys(stub, {'dark_factory': '/repo/df'}, keys)

        assert result.corroborated_keys == {('dark_factory', '1')}
        assert (result.corroborated, result.uncorroborated, result.unresolvable) == (1, 1, 1)
        assert _totals_account_for_every_key(result, keys)

    @pytest.mark.asyncio
    async def test_all_lookups_are_dispatched_as_one_concurrent_batch(self):
        """One flat gather, not a sequential await loop: peak in-flight
        lookups must equal the number of resolvable keys."""
        stub = _StubTaskmaster()
        keys = {('dark_factory', str(i)) for i in range(5)}

        await _corroborate_record_keys(stub, {'dark_factory': '/repo/df'}, keys)

        assert len(stub.calls) == 5
        assert stub.max_concurrent == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'taskmaster_arg, known_projects_arg',
        [
            (None, {'dark_factory': '/repo/df'}),
            (False, {'dark_factory': '/repo/df'}),
            ('STUB', None),
            ('STUB', {}),
        ],
        ids=['taskmaster-none', 'taskmaster-false', 'known-projects-none', 'known-projects-empty'],
    )
    async def test_falsy_taskmaster_or_known_projects_short_circuits(
        self, taskmaster_arg, known_projects_arg,
    ):
        """Corroboration cannot run at all: nothing corroborated, every key
        counted UNRESOLVABLE so the withheld repair stays visible, no I/O."""
        stub = _StubTaskmaster()
        taskmaster = stub if taskmaster_arg == 'STUB' else taskmaster_arg
        keys = {('dark_factory', '3045'), ('reify', '9')}

        result = await _corroborate_record_keys(taskmaster, known_projects_arg, keys)

        assert result.corroborated == 0
        assert result.corroborated_keys == set()
        assert result.uncorroborated == 0
        assert result.unresolvable == 2
        assert _totals_account_for_every_key(result, keys)
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_empty_keys_is_a_no_op(self):
        stub = _StubTaskmaster()

        result = await _corroborate_record_keys(stub, {'dark_factory': '/repo/df'}, set())

        assert (result.corroborated, result.uncorroborated, result.unresolvable) == (0, 0, 0)
        assert result.corroborated_keys == set()
        assert stub.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'keys',
        [None, 'not-a-set', 42, {('dark_factory', None)}, {('dark_factory', 3045)}],
        ids=['none', 'str', 'int', 'none-task-id', 'int-task-id'],
    )
    async def test_never_raises_on_malformed_keys(self, keys):
        result = await _corroborate_record_keys(
            _StubTaskmaster(), {'dark_factory': '/repo/df'}, keys,
        )

        assert result.corroborated == 0

    @pytest.mark.asyncio
    async def test_exploding_known_projects_degrades_to_nothing_corroborated(self):
        """A defect in the corroboration pass itself must never abort an
        otherwise-good stage report — it degrades to 'we could not check'."""
        keys = {('dark_factory', '3045'), ('reify', '9')}

        result = await _corroborate_record_keys(
            _StubTaskmaster(), _ExplodingProjects(dark_factory='/repo/df'), keys,
        )

        assert result.corroborated == 0
        assert result.corroborated_keys == set()
        assert result.unresolvable == 2
        assert _totals_account_for_every_key(result, keys)


# ── step-7: the repair is driven by the CORROBORATED count ───────────────────

_UNDERCOUNT_EVENT = 'reconciliation.stage2_tasks_created_undercount'

_RUN_507BC25B_RECORD = {
    'action': 'task_created',
    'task_id': '3045',
    'status': 'created',
    'project_id': 'dark_factory',
    'source_path': 'proactive_sample',
}

_KNOWN_PROJECTS = {'dark_factory': '/repo/df'}


def _undercount_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and _UNDERCOUNT_EVENT in r.getMessage()
    ]


class TestPostFlightRepairIsCorroborated:
    """``_apply_post_flight_guards`` may only raise ``tasks_created`` on a
    record whose task ``get_task`` confirms EXISTS (task 3051).

    Task 3046 repaired the counter upward from ``task_created_records``, but
    that list is itself pure LLM self-report — a hallucinated record inflated
    the counter with no external check. The repair is now driven by the
    corroborated count instead.

    Exercised directly (not via ``stage.run()``): with ``flag_deleted_records``
    absent, ``_acknowledge_resolved_stage1_markers`` short-circuits to 0
    without touching the mocked memory service, isolating these assertions to
    the tasks_created path.
    """

    @pytest.mark.asyncio
    async def test_run_507bc25b_repairs_when_the_record_is_corroborated(self, caplog):
        """The exact run-507bc25b shape — tasks_created=0 self-reported while
        one record confirms task 3045 was filed — still repairs to 1, now that
        get_task confirms 3045 exists in dark_factory."""
        stub = _StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')})
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [dict(_RUN_507BC25B_RECORD)],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 1
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 1
        assert report.stats['task_created_records_corroborated'] == 1
        assert report.stats['task_created_records_uncorroborated'] == 0
        assert report.stats['task_created_records_unresolvable'] == 0
        assert stub.calls == [('3045', '/repo/df')]

        warnings = _undercount_warnings(caplog)
        assert len(warnings) == 1
        assert 'to 1' in warnings[0]

    @pytest.mark.asyncio
    async def test_uncorroborated_record_withholds_the_repair(self, caplog):
        """Same stats, but get_task says the cited task does not exist: the
        self-reported 0 stands. No repair, no tasks_created_reported key, and
        the counter is never moved DOWN."""
        stub = _StubTaskmaster(results={('3045', '/repo/df'): _not_found('3045')})
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [dict(_RUN_507BC25B_RECORD)],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 0
        assert 'tasks_created_reported' not in report.stats
        assert report.stats['task_created_records_valid'] == 1
        assert report.stats['task_created_records_corroborated'] == 0
        assert report.stats['task_created_records_uncorroborated'] == 1
        assert report.stats['task_created_records_unresolvable'] == 0
        assert _undercount_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_partially_corroborated_batch_repairs_to_the_corroborated_count(self):
        """Two records, one real and one invented: repairs to 1, not 2."""
        stub = _StubTaskmaster(results={
            ('3045', '/repo/df'): _task_record('3045'),
            ('9999', '/repo/df'): _not_found('9999'),
        })
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                dict(_RUN_507BC25B_RECORD),
                {'action': 'task_created', 'task_id': '9999', 'status': 'created',
                 'project_id': 'dark_factory'},
            ],
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 1
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 2
        assert report.stats['task_created_records_corroborated'] == 1
        assert report.stats['task_created_records_uncorroborated'] == 1

    @pytest.mark.asyncio
    async def test_corroboration_never_lowers_a_self_report(self, caplog):
        """Corroboration is fail-closed on the INCREMENT only: a self-report
        above the corroborated count survives untouched (the symmetric
        downward clamp task 2230 removed is not reintroduced here)."""
        stub = _StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')})
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 5,
            'task_created_records': [dict(_RUN_507BC25B_RECORD)],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 5
        assert 'tasks_created_reported' not in report.stats
        assert report.stats['task_created_records_corroborated'] == 1
        assert _undercount_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_observability_keys_are_always_published(self):
        """Even with no records at all, the four record stats are present so
        Stage 3's audit sees a deterministic set."""
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        report = _make_report({})

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert report.stats['task_created_records_corroborated'] == 0
        assert report.stats['task_created_records_uncorroborated'] == 0
        assert report.stats['task_created_records_unresolvable'] == 0

    @pytest.mark.asyncio
    async def test_records_valid_keeps_its_pre_corroboration_meaning(self):
        """task_created_records_valid is the STRUCTURALLY-valid record count —
        task 3045's lesson that a stat name must not imply a stronger
        guarantee than the code delivers. Corroboration is reported
        separately, never folded into it."""
        stub = _StubTaskmaster()  # every lookup not-found
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                {'task_id': '1', 'status': 'created', 'project_id': 'dark_factory'},
                {'task_id': '2', 'status': 'combined', 'project_id': 'dark_factory'},
                {'task_id': '3', 'status': 'failed', 'project_id': 'dark_factory'},
            ],
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        # 'failed' is not structurally valid; the other two are, and neither
        # corroborates.
        assert report.stats['task_created_records_valid'] == 2
        assert report.stats['task_created_records_corroborated'] == 0
        assert report.stats['task_created_records_uncorroborated'] == 2

    @pytest.mark.asyncio
    async def test_repair_warning_names_the_corroborated_count(self, caplog):
        """The undercount WARNING fires only on an actual repair and reports
        the CORROBORATED count — not the raw record count, which is what it
        would have overstated before task 3051."""
        stub = _StubTaskmaster(results={
            ('1', '/repo/df'): _task_record('1'),
            ('2', '/repo/df'): _task_record('2'),
            ('3', '/repo/df'): _not_found('3'),
        })
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                {'task_id': str(i), 'status': 'created', 'project_id': 'dark_factory'}
                for i in (1, 2, 3)
            ],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 2
        warnings = _undercount_warnings(caplog)
        assert len(warnings) == 1
        assert 'to 2' in warnings[0]
        assert 'test-run-3051' in warnings[0]
        assert _TEST_PROJECT_ID in warnings[0]

    @pytest.mark.asyncio
    async def test_a_cross_project_record_is_corroborated_in_its_own_project(self):
        """A record filed by Cross-Project Routing is checked against ITS
        project's root, not this stage's."""
        stub = _StubTaskmaster(results={('77', '/repo/reify'): _task_record('77')})
        stage = _make_stage(
            known_projects={'dark_factory': '/repo/df', 'reify': '/repo/reify'},
            taskmaster=stub,
        )
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                {'task_id': '77', 'status': 'created', 'project_id': 'reify'},
            ],
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 1
        assert stub.calls == [('77', '/repo/reify')]


# ── step-9: loud degradation when corroboration cannot run ───────────────────

_UNCORROBORATED_EVENT = 'reconciliation.stage2_task_created_records_uncorroborated'


def _uncorroborated_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and _UNCORROBORATED_EVENT in r.getMessage()
    ]


def _two_records() -> list[dict]:
    return [
        {'action': 'task_created', 'task_id': '3045', 'status': 'created',
         'project_id': 'dark_factory'},
        {'action': 'task_created', 'task_id': '3046', 'status': 'created',
         'project_id': 'dark_factory'},
    ]


class TestPostFlightLoudDegradation:
    """A withheld repair must be LOUD, never silent (task 3051).

    Fail-closed corroboration trades one invisible failure (an inflated
    counter) for another — a lost repair — unless the withholding is
    reported. Per the project's loud-over-silent-degradation norm and the
    no-silent-fail-soft design invariant, every configuration in which
    corroboration cannot run publishes the full record-stat split AND emits a
    WARNING distinct from the undercount-repair one, so an operator can tell
    "the agent invented records" from "we could not check".
    """

    def _degraded_stages(self):
        """The three ways corroboration can fail to confirm anything."""
        no_taskmaster = _make_stage(known_projects=_KNOWN_PROJECTS)
        no_taskmaster.taskmaster = None

        no_projects = _make_stage(taskmaster=_StubTaskmaster())
        assert no_projects.known_projects == {}

        all_raise = _make_stage(
            known_projects=_KNOWN_PROJECTS,
            taskmaster=_StubTaskmaster(default=RuntimeError('backend down')),
        )
        return {
            'no-taskmaster': no_taskmaster,
            'empty-known-projects': no_projects,
            'every-lookup-raises': all_raise,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'which', ['no-taskmaster', 'empty-known-projects', 'every-lookup-raises'],
    )
    async def test_no_repair_and_the_self_report_stands(self, which):
        stage = self._degraded_stages()[which]
        report = _make_report({
            'tasks_created': '1',
            'task_created_records': _two_records(),
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        # Coerced (str -> int) but never repaired upward from the records.
        assert report.stats['tasks_created'] == 1
        assert isinstance(report.stats['tasks_created'], int)
        assert 'tasks_created_reported' not in report.stats

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'which', ['no-taskmaster', 'empty-known-projects', 'every-lookup-raises'],
    )
    async def test_every_valid_record_is_accounted_for(self, which):
        """The withheld repair is visible: all four record stats are present
        and the three corroboration buckets partition the valid records."""
        stage = self._degraded_stages()[which]
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': _two_records(),
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        valid = report.stats['task_created_records_valid']
        assert valid == 2
        assert report.stats['task_created_records_corroborated'] == 0
        buckets = (
            report.stats['task_created_records_uncorroborated']
            + report.stats['task_created_records_unresolvable']
        )
        assert buckets == valid

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'which', ['no-taskmaster', 'empty-known-projects', 'every-lookup-raises'],
    )
    async def test_exactly_one_distinct_warning_names_the_run_and_the_shortfall(
        self, which, caplog,
    ):
        """One WARNING per cycle, under an event name distinct from
        stage2_tasks_created_undercount, carrying run_id, project_id and the
        number of records that could not be corroborated."""
        stage = self._degraded_stages()[which]
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': _two_records(),
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        warnings = _uncorroborated_warnings(caplog)
        assert len(warnings) == 1
        assert 'test-run-3051' in warnings[0]
        assert _TEST_PROJECT_ID in warnings[0]
        assert '2' in warnings[0]
        assert _undercount_warnings(caplog) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'which', ['no-taskmaster', 'empty-known-projects', 'every-lookup-raises'],
    )
    async def test_the_rest_of_the_guard_still_runs(self, which):
        """Degradation is confined to the tasks_created repair: the guard does
        not raise and still normalizes the flag counters and publishes the
        marker-acknowledgment count."""
        stage = self._degraded_stages()[which]
        report = _make_report({'task_created_records': _two_records()})

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['stage1_analytical_findings_processed'] == 0
        assert report.stats['stage1_mem0_flags_processed'] == 0
        assert report.stats['stage2_flag_markers_acknowledged'] == 0

    @pytest.mark.asyncio
    async def test_no_warning_when_there_is_nothing_to_corroborate(self, caplog):
        """The WARNING fires only when a structurally-valid record failed to
        corroborate — an empty/absent record list is not a degradation."""
        stage = _make_stage(known_projects=_KNOWN_PROJECTS)
        stage.taskmaster = None
        report = _make_report({'tasks_created': 0, 'task_created_records': []})

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert _uncorroborated_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_no_warning_when_every_record_corroborates(self, caplog):
        stub = _StubTaskmaster(results={
            ('3045', '/repo/df'): _task_record('3045'),
            ('3046', '/repo/df'): _task_record('3046'),
        })
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({'tasks_created': 2, 'task_created_records': _two_records()})

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['task_created_records_corroborated'] == 2
        assert _uncorroborated_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_partial_corroboration_repairs_and_warns_about_the_remainder(
        self, caplog,
    ):
        """Both WARNINGs can legitimately fire in one cycle: the repair moved
        up to what WAS confirmed, and the rest is reported as unconfirmed."""
        stub = _StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')})
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({'tasks_created': 0, 'task_created_records': _two_records()})

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['tasks_created'] == 1
        assert len(_undercount_warnings(caplog)) == 1
        assert len(_uncorroborated_warnings(caplog)) == 1


class TestPostFlightFlagCountersAreNeverClamped:
    """Audit conclusion for the adjacent flag counters, as executable behaviour.

    Framework clamps once checked ``stage1_mem0_flags_processed`` against
    ``flag_deleted_records`` and ``stage1_analytical_findings_processed``
    against ``len(prior_reports[0].items_flagged)``, and ``prompts/stage2.py``
    never told the agent they were gone. Neither clamp has existed since
    tasks 2229/2230 (W5-mu) — ``_apply_post_flight_guards`` only
    ``setdefault``-normalizes both keys. These pins keep that true (and are
    what makes the step-11 prompt correction verifiable in code rather than
    prose).
    """

    @pytest.mark.asyncio
    async def test_self_reported_flag_counters_survive_untouched(self):
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        prior = _make_report({})
        prior.items_flagged = [{'id': str(i)} for i in range(4)]
        report = _make_report({
            'stage1_analytical_findings_processed': 1,
            'stage1_mem0_flags_processed': 5,
            'flag_deleted_records': [
                {'action': 'flag_deleted', 'flag_id': f'flag-{i}'} for i in range(2)
            ],
        })

        await stage._apply_post_flight_guards(report, [prior], 'test-run-3051')

        assert report.stats['stage1_analytical_findings_processed'] == 1
        assert report.stats['stage1_mem0_flags_processed'] == 5

    @pytest.mark.asyncio
    async def test_absent_flag_counters_default_to_zero(self):
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        report = _make_report({})

        await stage._apply_post_flight_guards(report, [], 'test-run-3051')

        assert report.stats['stage1_analytical_findings_processed'] == 0
        assert report.stats['stage1_mem0_flags_processed'] == 0
