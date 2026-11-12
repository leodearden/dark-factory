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

**Task counters — grounded by ``stage2_counter_grounding`` (task 4873).**
``derive_stage_stats`` does not recompute these; instead
``TaskKnowledgeSync._apply_post_flight_guards`` repairs each one UPWARD to
its confirmed evidence:

1. ``tasks_created`` / ``tasks_combined`` — evidence is this run's creation
   journal (``middleware.task_creation_journal``: one row per curator outcome,
   written by ``TaskInterceptor`` against the ``_causation_id`` in
   ``submit_task`` metadata, trusted as code truth) united with the
   ``task_created_records`` claims of status ``created`` / ``combined``, each
   claim confirmed with ``taskmaster.get_task`` in its own project's root
   (task 3051). A journal row also binds the task to this run; a confirmed
   record proves only that the task exists.
2. ``tasks_hints_updated`` — evidence is the ``task_hints_updated_records``
   claims, each confirmed when the re-read task's ``memory_hints`` are a
   SUPERSET of the recorded entries (task 4018).
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
``<prefix>_uncorroborated`` / ``_unresolvable`` stats).

Per the ``test_recon_gate_closure_guidance.py`` house rule, prompt prose is
intentionally NOT pinned here — only machine-readable contract tokens and
code behaviour.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stage2_counter_grounding import (
    CounterEvidence,
    CounterGrounding,
    HintsClaim,
    action_record_keys,
    ground_counter,
    hints_claim_confirmed,
    hints_update_claims,
)
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync
from fused_memory.reconciliation.task_corroboration import confirm_task_present

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
    journal_rows: list[dict] | None = None,
    project_id: str = _TEST_PROJECT_ID,
) -> dict:
    """Kwargs for constructing a TaskKnowledgeSync with mocked deps.

    *journal_rows* is what the write journal's ``get_ops_by_causation``
    returns — rows shaped like ``WriteJournal`` output (see ``_journal_row``).
    """
    config = ReconciliationConfig(enabled=True, explore_codebase_root=_TEST_PROJECT_ROOT)
    write_journal_mock = MagicMock()
    write_journal_mock.get_ops_by_causation = AsyncMock(return_value=list(journal_rows or []))
    journal_mock = MagicMock()
    journal_mock.write_journal = write_journal_mock
    journal_mock.record_run_session = AsyncMock()
    journal_mock.clear_run_session = AsyncMock()
    return {
        'memory_service': AsyncMock(),
        'taskmaster': AsyncMock() if taskmaster is None else taskmaster,
        'journal': journal_mock,
        'config': config,
        'scope': ProjectScope(ProjectId(project_id), ProjectRoot(_TEST_PROJECT_ROOT)),
        'known_projects': dict(known_projects) if known_projects else {},
    }


def _make_stage(
    known_projects: dict[str, str] | None = None,
    taskmaster: Any = None,
    journal_rows: list[dict] | None = None,
    project_id: str = _TEST_PROJECT_ID,
) -> TaskKnowledgeSync:
    return TaskKnowledgeSync(
        StageId.task_knowledge_sync,
        **_mock_deps(
            known_projects=known_projects,
            taskmaster=taskmaster,
            journal_rows=journal_rows,
            project_id=project_id,
        ),
    )


def _journal_row(
    run_id: str,
    task_id: object,
    *,
    project_id: str | None,
    operation: str = 'task_created',
    success: int = 1,
) -> dict:
    """A creation-journal write_op row as ``WriteJournal.get_ops_by_causation`` returns it."""
    return {
        'layer': 'write_op',
        'operation': operation,
        'causation_id': run_id,
        'project_id': project_id,
        'agent_id': 'recon-stage-task_knowledge_sync',
        'params': json.dumps({'task_id': task_id}),
        'success': success,
    }


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


# ── action_record_keys ───────────────────────────────────────────────────────

_CREATED = frozenset({'created'})
_COMBINED = frozenset({'combined'})


class _ExplodingRecord(dict):
    """A dict-shaped record that raises while being inspected."""

    def get(self, key, default=None):  # type: ignore[override]
        raise ValueError(f'record inspection exploded on {key!r}')


class TestActionRecordKeys:
    """``action_record_keys(records, default_project_id, valid_statuses)``.

    Returns the deduped ``(project_id, task_id)`` keys of an LLM action-record
    list, so each key can be corroborated against its OWN project. Every rule
    task 3046 established is pinned here. ``valid_statuses`` is REQUIRED: the
    created and combined counters read the same list with disjoint vocabularies,
    so no default can be right for both.
    """

    def test_returns_a_frozenset_of_project_task_pairs(self):
        keys = action_record_keys(
            [
                {'action': 'task_created', 'task_id': '3045', 'status': 'created',
                 'project_id': 'dark_factory'},
            ],
            None,
            _CREATED,
        )

        assert keys == {('dark_factory', '3045')}
        assert isinstance(keys, frozenset)

    @pytest.mark.parametrize('status', ['created', 'CREATED', ' Created ', '\tcreated\n'])
    def test_status_match_is_case_and_whitespace_insensitive(self, status):
        keys = action_record_keys(
            [{'task_id': '1', 'status': status, 'project_id': 'p'}], None, _CREATED,
        )

        assert keys == {('p', '1')}

    @pytest.mark.parametrize('status', ['failed', 'FAILED', ' failed ', 'pending', '', None, 7])
    def test_non_accepted_status_never_keys_even_with_a_task_id(self, status):
        """`failed` is never counted regardless of a present task_id."""
        keys = action_record_keys(
            [{'task_id': '3045', 'status': status, 'project_id': 'p'}], None, _CREATED,
        )

        assert keys == frozenset()

    def test_missing_status_key_never_keys(self):
        keys = action_record_keys(
            [{'task_id': '3045', 'project_id': 'p'}], None, _CREATED,
        )

        assert keys == frozenset()

    def test_created_vocabulary_never_keys_a_combined_record(self):
        records = [{'task_id': '1', 'status': 'combined', 'project_id': 'p'}]

        assert action_record_keys(records, 'p', _CREATED) == frozenset()

    def test_combined_vocabulary_never_keys_a_created_record(self):
        records = [{'task_id': '1', 'status': 'created', 'project_id': 'p'}]

        assert action_record_keys(records, 'p', _COMBINED) == frozenset()

    def test_the_two_vocabularies_partition_a_mixed_list(self):
        records = [
            {'task_id': '1', 'status': 'created', 'project_id': 'p'},
            {'task_id': '2', 'status': 'combined', 'project_id': 'p'},
        ]

        assert action_record_keys(records, 'p', _CREATED) == {('p', '1')}
        assert action_record_keys(records, 'p', _COMBINED) == {('p', '2')}

    @pytest.mark.parametrize('task_id', [None, '', '   '])
    def test_missing_or_blank_task_id_is_skipped(self, task_id):
        keys = action_record_keys(
            [{'task_id': task_id, 'status': 'created', 'project_id': 'p'}], None, _CREATED,
        )

        assert keys == frozenset()

    def test_absent_task_id_key_is_skipped(self):
        keys = action_record_keys(
            [{'status': 'created', 'project_id': 'p'}], None, _CREATED,
        )

        assert keys == frozenset()

    def test_same_numeric_id_under_two_projects_is_two_keys(self):
        """Taskmaster ids are per-project, so cross-project routing filing id
        3045 into two projects is TWO tasks, not one duplicate report."""
        keys = action_record_keys(
            [
                {'task_id': 3045, 'status': 'created', 'project_id': 'dark_factory'},
                {'task_id': '3045', 'status': 'created', 'project_id': 'reify'},
            ],
            None,
            _CREATED,
        )

        assert keys == {('dark_factory', '3045'), ('reify', '3045')}

    def test_int_and_str_task_id_forms_collapse_to_one_key(self):
        keys = action_record_keys(
            [
                {'task_id': 3045, 'status': 'created', 'project_id': 'dark_factory'},
                {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
                {'task_id': ' 3045 ', 'status': 'created', 'project_id': 'dark_factory'},
            ],
            None,
            _CREATED,
        )

        assert keys == {('dark_factory', '3045')}

    def test_whitespace_padded_project_id_collapses_to_one_key(self):
        keys = action_record_keys(
            [
                {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
                {'task_id': '3045', 'status': 'created', 'project_id': ' dark_factory '},
            ],
            None,
            _CREATED,
        )

        assert keys == {('dark_factory', '3045')}

    @pytest.mark.parametrize('project_id', [None, '', '   '])
    def test_missing_or_blank_project_id_falls_back_to_default(self, project_id):
        """An omitted project_id must not masquerade as a second, distinct
        cross-project filing of the same task."""
        keys = action_record_keys(
            [
                {'task_id': '3045', 'status': 'created', 'project_id': project_id},
                {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            ],
            'dark_factory',
            _CREATED,
        )

        assert keys == {('dark_factory', '3045')}

    def test_absent_project_id_key_falls_back_to_default(self):
        keys = action_record_keys(
            [{'task_id': '3045', 'status': 'created'}], 'dark_factory', _CREATED,
        )

        assert keys == {('dark_factory', '3045')}

    def test_none_default_keys_an_unattributed_record_as_none(self):
        keys = action_record_keys([{'task_id': '3045', 'status': 'created'}], None, _CREATED)

        assert keys == {(None, '3045')}

    @pytest.mark.parametrize(
        'records',
        [None, 'not-a-list', 42, {}, {'task_id': '1'}, [], ()],
        ids=['none', 'str', 'int', 'empty-dict', 'dict', 'empty-list', 'tuple'],
    )
    def test_non_list_or_empty_input_returns_empty_set(self, records):
        assert action_record_keys(records, None, _CREATED) == frozenset()

    def test_non_dict_and_exploding_entries_are_skipped_not_propagated(self):
        """A malformed record degrades to 'not counted', never to an exception
        that would corrupt an otherwise-good stage report."""
        keys = action_record_keys(
            [
                None,
                'x',
                ['a'],
                _ExplodingRecord(task_id='9', status='created', project_id='p'),
                {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            ],
            None,
            _CREATED,
        )

        assert keys == {('dark_factory', '3045')}


# ── hints_update_claims / hints_claim_confirmed (task 4018) ─────────────────


def _hints_record(
    task_id: object = '7',
    project_id: object = 'dark_factory',
    entities=None,
    queries=None,
    **extra,
):
    record = {
        'action': 'task_hints_updated',
        'task_id': task_id,
        'project_id': project_id,
        'entities': ['EntityA'] if entities is None else entities,
        'queries': ['how is A wired'] if queries is None else queries,
    }
    record.update(extra)
    return record


def _task_with_hints(hints: object, *, as_json_string: bool = False) -> dict:
    metadata: object = {'memory_hints': hints}
    if as_json_string:
        metadata = json.dumps(metadata)
    return {'id': '7', 'title': 'A real task', 'status': 'pending', 'metadata': metadata}


class TestHintsUpdateClaims:
    """``hints_update_claims`` reads Stage 2's ``task_hints_updated_records``;
    ``hints_claim_confirmed`` decides whether a re-read task really carries
    the claimed hints (the SUPERSET rule the Stage 2 prompt already states)."""

    def test_one_record_is_one_claim_keyed_like_action_records(self):
        claims = hints_update_claims([_hints_record(task_id=' 7 ', project_id=' dark_factory ')], 'x')

        assert claims == {
            ('dark_factory', '7'): HintsClaim(
                entities=frozenset({'EntityA'}), queries=frozenset({'how is A wired'}),
            ),
        }

    @pytest.mark.parametrize('project_id', [None, '', '   '])
    def test_missing_project_id_falls_back_to_default(self, project_id):
        claims = hints_update_claims([_hints_record(project_id=project_id)], 'dark_factory')

        assert set(claims) == {('dark_factory', '7')}

    def test_duplicate_records_for_one_key_union_their_entries(self):
        claims = hints_update_claims(
            [
                _hints_record(entities=['A'], queries=['q1']),
                _hints_record(task_id=7, entities=['B'], queries=[]),
            ],
            'dark_factory',
        )

        assert claims == {
            ('dark_factory', '7'): HintsClaim(
                entities=frozenset({'A', 'B'}), queries=frozenset({'q1'}),
            ),
        }

    @pytest.mark.parametrize(
        'record',
        [
            _hints_record(entities=[], queries=[]),
            {'action': 'task_hints_updated', 'task_id': '7', 'project_id': 'p'},
            _hints_record(action='task_created'),
            _hints_record(action=None),
            _hints_record(task_id=None),
            _hints_record(task_id='  '),
        ],
        ids=['both-empty', 'both-absent', 'wrong-action', 'no-action', 'no-task-id', 'blank-task-id'],
    )
    def test_non_claims_are_skipped(self, record):
        assert hints_update_claims([record], 'p') == {}

    def test_non_str_entries_compare_by_canonical_json(self):
        claims = hints_update_claims(
            [_hints_record(entities=[{'b': 1, 'a': 2}], queries=['q'])], 'p',
        )

        claim = claims[('dark_factory', '7')]
        assert claim.entities == frozenset({json.dumps({'a': 2, 'b': 1}, sort_keys=True)})

    @pytest.mark.parametrize(
        'records',
        [None, 'x', 42, {}, [], ()],
        ids=['none', 'str', 'int', 'dict', 'empty-list', 'tuple'],
    )
    def test_non_list_input_is_no_claims(self, records):
        assert hints_update_claims(records, 'p') == {}

    def test_non_dict_and_exploding_entries_are_skipped_not_raised(self):
        claims = hints_update_claims(
            [None, 'x', _ExplodingRecord(task_id='9'), _hints_record()], 'p',
        )

        assert set(claims) == {('dark_factory', '7')}

    def test_claim_is_frozen(self):
        claim = HintsClaim(entities=frozenset({'A'}), queries=frozenset())

        for field in dataclasses.fields(claim):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(claim, field.name, None)

    # -- hints_claim_confirmed ------------------------------------------------

    _CLAIM = HintsClaim(entities=frozenset({'A'}), queries=frozenset({'q1'}))

    @pytest.mark.parametrize('as_json_string', [False, True], ids=['dict', 'json-string'])
    def test_superset_hints_confirm(self, as_json_string):
        result = _task_with_hints(
            {'entities': ['A', 'Other'], 'queries': ['q1', 'q2']}, as_json_string=as_json_string,
        )

        assert hints_claim_confirmed(self._CLAIM, result) is True

    def test_exactly_equal_hints_confirm(self):
        result = _task_with_hints({'entities': ['A'], 'queries': ['q1']})

        assert hints_claim_confirmed(self._CLAIM, result) is True

    def test_non_str_entries_confirm_by_canonical_json(self):
        claim = HintsClaim(
            entities=frozenset({json.dumps({'a': 2, 'b': 1}, sort_keys=True)}), queries=frozenset(),
        )
        result = _task_with_hints({'entities': [{'b': 1, 'a': 2}], 'queries': []})

        assert hints_claim_confirmed(claim, result) is True

    @pytest.mark.parametrize(
        'result',
        [
            _task_with_hints({'entities': ['A'], 'queries': []}),
            _task_with_hints({'entities': [], 'queries': ['q1']}),
            _task_with_hints({}),
            _task_with_hints(None),
            _task_with_hints(['A', 'q1']),
            {'id': '7', 'title': 'A real task', 'status': 'pending'},
            {'id': '7', 'title': 'A real task', 'status': 'pending', 'metadata': 'not json'},
            {'id': '7', 'title': 'A real task', 'status': 'pending', 'metadata': '[1, 2]'},
            {'error': 'No tasks found for ID(s): 7', 'error_type': 'TaskmasterError'},
            None,
            'a task',
        ],
        ids=[
            'missing-query', 'missing-entity', 'empty-hints', 'none-hints', 'legacy-list-hints',
            'no-metadata', 'undecodable-metadata', 'non-object-metadata', 'not-found', 'none', 'str',
        ],
    )
    def test_anything_short_of_a_superset_is_unconfirmed(self, result):
        assert hints_claim_confirmed(self._CLAIM, result) is False


# ── ground_counter ───────────────────────────────────────────────────────────
#
# The resolve/lookup/classify loop itself is pinned in
# tests/test_flag_dedup.py::TestCorroborateTaskLookups; these pins cover what
# ground_counter adds on top: journal trust and the partition of the rest.


def _present_claims(*keys) -> dict:
    return dict.fromkeys(keys, confirm_task_present)


class TestGroundCounter:
    """``ground_counter(taskmaster, known_projects, evidence)``."""

    @pytest.mark.asyncio
    async def test_journal_keys_count_and_are_never_looked_up(self):
        stub = _StubTaskmaster()
        evidence = CounterEvidence(
            journaled=frozenset({('dark_factory', '1'), ('reify', '2')}), claims={},
        )

        grounding = await ground_counter(stub, {}, evidence)

        assert grounding == CounterGrounding(
            journaled=2, record_confirmed=0, uncorroborated=0, unresolvable=0,
        )
        assert grounding.valid == 2
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_claims_the_journal_covers_are_dropped_before_lookup(self):
        stub = _StubTaskmaster(results={('2', '/repo/df'): _task_record('2')})
        evidence = CounterEvidence(
            journaled=frozenset({('dark_factory', '1')}),
            claims=_present_claims(('dark_factory', '1'), ('dark_factory', '2')),
        )

        grounding = await ground_counter(stub, _KNOWN_PROJECTS, evidence)

        assert (grounding.journaled, grounding.record_confirmed) == (1, 1)
        assert grounding.valid == 2
        assert stub.calls == [('2', '/repo/df')]

    @pytest.mark.asyncio
    async def test_record_only_claims_are_partitioned_by_verdict(self):
        stub = _StubTaskmaster(results={
            ('1', '/repo/df'): _task_record('1'),
            ('2', '/repo/df'): _not_found('2'),
        })
        evidence = CounterEvidence(
            journaled=frozenset(),
            claims=_present_claims(
                ('dark_factory', '1'), ('dark_factory', '2'), ('nowhere', '3'),
            ),
        )

        grounding = await ground_counter(stub, _KNOWN_PROJECTS, evidence)

        assert grounding == CounterGrounding(
            journaled=0, record_confirmed=1, uncorroborated=1, unresolvable=1,
        )

    @pytest.mark.asyncio
    async def test_each_claim_is_confirmed_by_its_own_predicate(self):
        stub = _StubTaskmaster(results={
            ('1', '/repo/df'): _task_record('1', title='Right'),
            ('2', '/repo/df'): _task_record('2', title='Wrong'),
        })

        def titled_right(result: object) -> bool:
            return isinstance(result, dict) and result.get('title') == 'Right'

        evidence = CounterEvidence(
            journaled=frozenset(),
            claims={('dark_factory', '1'): titled_right, ('dark_factory', '2'): titled_right},
        )

        grounding = await ground_counter(stub, _KNOWN_PROJECTS, evidence)

        assert (grounding.record_confirmed, grounding.uncorroborated) == (1, 1)

    @pytest.mark.asyncio
    async def test_no_taskmaster_leaves_record_claims_unresolvable_but_trusts_the_journal(self):
        evidence = CounterEvidence(
            journaled=frozenset({('dark_factory', '9')}),
            claims=_present_claims(('dark_factory', '1'), ('dark_factory', '2')),
        )

        grounding = await ground_counter(None, _KNOWN_PROJECTS, evidence)

        assert grounding == CounterGrounding(
            journaled=1, record_confirmed=0, uncorroborated=0, unresolvable=2,
        )

    def test_grounding_is_frozen(self):
        grounding = CounterGrounding(
            journaled=0, record_confirmed=0, uncorroborated=0, unresolvable=0,
        )

        for field in dataclasses.fields(grounding):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(grounding, field.name, None)


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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['tasks_created'] == 1
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 1
        assert report.stats['task_created_records_journaled'] == 0
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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['tasks_created'] == 0
        assert 'tasks_created_reported' not in report.stats
        assert report.stats['task_created_records_valid'] == 0
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

        await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['tasks_created'] == 1
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 1
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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['tasks_created'] == 5
        assert 'tasks_created_reported' not in report.stats
        assert report.stats['task_created_records_valid'] == 1
        assert _undercount_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_observability_keys_are_always_published(self):
        """Even with no records at all, the four record stats are present so
        Stage 3's audit sees a deterministic set."""
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        report = _make_report({})

        await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert report.stats['task_created_records_journaled'] == 0
        assert report.stats['task_created_records_uncorroborated'] == 0
        assert report.stats['task_created_records_unresolvable'] == 0

    @pytest.mark.asyncio
    async def test_records_valid_counts_only_confirmed_creations(self):
        """task_created_records_valid counts CONFIRMED creations only (task
        4873): a record get_task cannot confirm is never valid, so the
        valid=0-despite-a-real-task discrepancy and its inverse cannot recur."""
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

        await stage._apply_post_flight_guards(report, 'test-run-3051')

        # 'failed' never counts and 'combined' is not a tasks_created status;
        # the one 'created' record does not corroborate, so nothing is valid.
        assert report.stats['task_created_records_valid'] == 0
        assert report.stats['task_created_records_uncorroborated'] == 1

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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

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

        await stage._apply_post_flight_guards(report, 'test-run-3051')

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

        await stage._apply_post_flight_guards(report, 'test-run-3051')

        # Coerced (str -> int) but never repaired upward from the records.
        assert report.stats['tasks_created'] == 1
        assert isinstance(report.stats['tasks_created'], int)
        assert 'tasks_created_reported' not in report.stats

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'which', ['no-taskmaster', 'empty-known-projects', 'every-lookup-raises'],
    )
    async def test_every_valid_record_is_accounted_for(self, which):
        """The withheld repair is visible: the record stats are present, nothing
        is valid, and the two shortfall buckets account for every record."""
        stage = self._degraded_stages()[which]
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': _two_records(),
        })

        await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['task_created_records_valid'] == 0
        buckets = (
            report.stats['task_created_records_uncorroborated']
            + report.stats['task_created_records_unresolvable']
        )
        assert buckets == 2

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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

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

        await stage._apply_post_flight_guards(report, 'test-run-3051')

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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['task_created_records_valid'] == 2
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
            await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['tasks_created'] == 1
        assert len(_undercount_warnings(caplog)) == 1
        assert len(_uncorroborated_warnings(caplog)) == 1


# ── task 4873: tasks_created is grounded in the creation journal ─────────────

_JOURNAL_UNAVAILABLE_EVENT = 'reconciliation.stage2_creation_journal_unavailable'

RUN_F9E49AEF = 'f9e49aef-1111-4000-8000-000000000105'
RUN_90D762D0 = '90d762d0-2222-4000-8000-000000004745'
RUN_507BC25B = '507bc25b-3333-4000-8000-000000003045'


def _warnings_named(caplog, event: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and event in r.getMessage()
    ]


class TestPostFlightJournalGrounding:
    """``tasks_created`` is repaired from the creation journal (task 4873).

    TaskInterceptor writes one row per creation outcome attributed to the
    submitter's ``_causation_id``; Stage 2's post-flight reads the run's rows.
    Journal keys are code truth — trusted, never re-looked-up. Records remain a
    corroborated second channel for creations the journal did not attribute.
    """

    @pytest.mark.asyncio
    async def test_run_f9e49aef_journal_confirms_the_self_report(self):
        """Task 105 filed with no task_created_records entry: the journal row
        alone makes it a confirmed creation."""
        stage = _make_stage(
            project_id='solar_challenge',
            journal_rows=[_journal_row(RUN_F9E49AEF, '105', project_id='solar_challenge')],
        )
        report = _make_report({'tasks_created': 1})

        await stage._apply_post_flight_guards(report, RUN_F9E49AEF)

        assert report.stats['task_created_records_valid'] == 1
        assert report.stats['task_created_records_journaled'] == 1
        assert report.stats['tasks_created'] == 1
        assert 'tasks_created_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_run_90d762d0_journal_repairs_an_omitted_creation(self, caplog):
        """Task 4745 filed via resolve_ticket, self-report 0, no record."""
        stage = _make_stage(
            project_id='dark_factory',
            journal_rows=[_journal_row(RUN_90D762D0, 4745, project_id='dark_factory')],
        )
        report = _make_report({'tasks_created': 0})

        with caplog.at_level(logging.WARNING):
            await stage._apply_post_flight_guards(report, RUN_90D762D0)

        assert report.stats['tasks_created'] == 1
        assert report.stats['tasks_created_reported'] == 0
        assert len(_warnings_named(caplog, _UNDERCOUNT_EVENT)) == 1

    @pytest.mark.asyncio
    async def test_journal_keys_are_trusted_without_a_lookup(self):
        stub = _StubTaskmaster()
        stage = _make_stage(
            taskmaster=stub,
            known_projects={},
            journal_rows=[_journal_row(RUN_90D762D0, 4745, project_id='dark_factory')],
        )
        report = _make_report({'tasks_created': 0})

        await stage._apply_post_flight_guards(report, RUN_90D762D0)

        assert report.stats['tasks_created'] == 1
        assert report.stats['task_created_records_unresolvable'] == 0
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_a_record_duplicating_a_journal_key_is_neither_counted_twice_nor_looked_up(
        self,
    ):
        stub = _StubTaskmaster()
        stage = _make_stage(
            taskmaster=stub,
            known_projects=_KNOWN_PROJECTS,
            journal_rows=[_journal_row(RUN_90D762D0, '4745', project_id='dark_factory')],
        )
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                {'action': 'task_created', 'task_id': 4745, 'status': 'created',
                 'project_id': 'dark_factory'},
            ],
        })

        await stage._apply_post_flight_guards(report, RUN_90D762D0)

        assert report.stats['tasks_created'] == 1
        assert report.stats['task_created_records_valid'] == 1
        assert report.stats['task_created_records_journaled'] == 1
        assert report.stats['task_created_records_uncorroborated'] == 0
        assert report.stats['task_created_records_unresolvable'] == 0
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_project_less_row_and_record_for_one_task_count_once(self):
        """Both channels key a missing project under Stage 2's own project."""
        stub = _StubTaskmaster()
        stage = _make_stage(
            taskmaster=stub,
            known_projects=_KNOWN_PROJECTS,
            project_id='dark_factory',
            journal_rows=[_journal_row(RUN_90D762D0, '4745', project_id=None)],
        )
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [{'task_id': ' 4745 ', 'status': 'created'}],
        })

        await stage._apply_post_flight_guards(report, RUN_90D762D0)

        assert report.stats['tasks_created'] == 1
        assert report.stats['task_created_records_valid'] == 1
        assert report.stats['task_created_records_journaled'] == 1
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_run_507bc25b_record_only_creation_still_repairs_via_get_task(self):
        stub = _StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')})
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [dict(_RUN_507BC25B_RECORD)],
        })

        await stage._apply_post_flight_guards(report, RUN_507BC25B)

        assert report.stats['tasks_created'] == 1
        assert report.stats['task_created_records_valid'] == 1
        assert report.stats['task_created_records_journaled'] == 0
        assert stub.calls == [('3045', '/repo/df')]

    @pytest.mark.asyncio
    async def test_stats_partition_journaled_and_record_only_claims(self):
        stub = _StubTaskmaster(results={
            ('2', '/repo/df'): _task_record('2'),
            ('3', '/repo/df'): _not_found('3'),
        })
        stage = _make_stage(
            taskmaster=stub,
            known_projects=_KNOWN_PROJECTS,
            journal_rows=[_journal_row(RUN_90D762D0, '1', project_id='dark_factory')],
        )
        records = [
            {'task_id': tid, 'status': 'created', 'project_id': pid}
            for tid, pid in [('1', 'dark_factory'), ('2', 'dark_factory'),
                             ('3', 'dark_factory'), ('4', 'nowhere')]
        ]
        report = _make_report({'tasks_created': 0, 'task_created_records': records})

        await stage._apply_post_flight_guards(report, RUN_90D762D0)

        stats = report.stats
        record_only_confirmed = stats['task_created_records_valid'] - stats[
            'task_created_records_journaled'
        ]
        assert stats['task_created_records_journaled'] == 1
        assert record_only_confirmed == 1
        assert stats['task_created_records_uncorroborated'] == 1
        assert stats['task_created_records_unresolvable'] == 1
        assert (
            record_only_confirmed
            + stats['task_created_records_uncorroborated']
            + stats['task_created_records_unresolvable']
        ) == 3
        assert stats['tasks_created'] == 2

    @pytest.mark.asyncio
    async def test_the_corroborated_key_is_no_longer_published(self):
        stage = _make_stage(
            known_projects=_KNOWN_PROJECTS,
            taskmaster=_StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')}),
        )
        report = _make_report({'task_created_records': [dict(_RUN_507BC25B_RECORD)]})

        await stage._apply_post_flight_guards(report, RUN_507BC25B)

        assert 'task_created_records_corroborated' not in report.stats

    @staticmethod
    def _unavailable_stage(which: str) -> TaskKnowledgeSync:
        deps = _mock_deps(
            known_projects=_KNOWN_PROJECTS,
            taskmaster=_StubTaskmaster(results={('3045', '/repo/df'): _task_record('3045')}),
        )
        journal = deps['journal']
        if which == 'no-write-journal':
            journal.write_journal = None
        elif which == 'read-raises':
            journal.write_journal.get_ops_by_causation = AsyncMock(
                side_effect=RuntimeError('db locked'),
            )
        else:
            journal.write_journal.get_ops_by_causation = AsyncMock(return_value=MagicMock())
        return TaskKnowledgeSync(StageId.task_knowledge_sync, **deps)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('which', ['no-write-journal', 'read-raises', 'non-list'])
    async def test_unavailable_journal_falls_back_to_records_and_warns_once(
        self, which, caplog,
    ):
        stage = self._unavailable_stage(which)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [dict(_RUN_507BC25B_RECORD)],
        })

        with caplog.at_level(logging.WARNING):
            await stage._apply_post_flight_guards(report, RUN_507BC25B)

        assert report.stats['tasks_created'] == 1
        assert report.stats['task_created_records_journaled'] == 0
        warnings = _warnings_named(caplog, _JOURNAL_UNAVAILABLE_EVENT)
        assert len(warnings) == 1
        assert RUN_507BC25B in warnings[0]
        assert _TEST_PROJECT_ID in warnings[0]

    @pytest.mark.asyncio
    async def test_an_available_journal_does_not_warn(self, caplog):
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        report = _make_report({})

        with caplog.at_level(logging.WARNING):
            await stage._apply_post_flight_guards(report, RUN_507BC25B)

        assert _warnings_named(caplog, _JOURNAL_UNAVAILABLE_EVENT) == []

    @pytest.mark.asyncio
    async def test_a_larger_self_report_is_never_lowered(self, caplog):
        stage = _make_stage(
            journal_rows=[_journal_row(RUN_90D762D0, '4745', project_id='dark_factory')],
        )
        report = _make_report({'tasks_created': 5})

        with caplog.at_level(logging.WARNING):
            await stage._apply_post_flight_guards(report, RUN_90D762D0)

        assert report.stats['tasks_created'] == 5
        assert 'tasks_created_reported' not in report.stats
        assert _warnings_named(caplog, _UNDERCOUNT_EVENT) == []


# ── task 4873 scope (2): combined outcomes count under tasks_combined ────────

RUN_COMBINED = 'c0mb1ned-4444-4000-8000-000000000010'


class TestPostFlightTasksCombinedSplit:
    """A curator fold (drop / combine / idempotency / candidate-key collision)
    mints no task id, so it counts under ``tasks_combined``, never
    ``tasks_created`` — the hole the 3051 review named."""

    @pytest.mark.asyncio
    async def test_a_combined_record_raises_tasks_combined_not_tasks_created(self):
        stub = _StubTaskmaster(results={('42', '/repo/df'): _task_record('42')})
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                {'action': 'task_created', 'task_id': '42', 'status': 'combined',
                 'project_id': 'dark_factory'},
            ],
        })

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        assert report.stats['tasks_created'] == 0
        assert report.stats['tasks_combined'] == 1
        assert report.stats['tasks_combined_evidence_valid'] == 1

    @pytest.mark.asyncio
    async def test_a_journal_combined_row_raises_tasks_combined_only(self):
        stage = _make_stage(journal_rows=[
            _journal_row(RUN_COMBINED, '42', project_id='dark_factory', operation='task_combined'),
        ])
        report = _make_report({})

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        assert report.stats['tasks_created'] == 0
        assert report.stats['tasks_combined'] == 1
        assert report.stats['tasks_combined_evidence_journaled'] == 1
        assert report.stats['task_created_records_journaled'] == 0

    @pytest.mark.asyncio
    async def test_mixed_batch_splits_created_from_combined(self):
        """Journal: 10 created, and a within-batch drop folded into that same
        new task 10; plus a corroborated 'created' record for 11."""
        stub = _StubTaskmaster(results={('11', '/repo/df'): _task_record('11')})
        stage = _make_stage(
            known_projects=_KNOWN_PROJECTS,
            taskmaster=stub,
            journal_rows=[
                _journal_row(RUN_COMBINED, '10', project_id='dark_factory'),
                _journal_row(
                    RUN_COMBINED, '10', project_id='dark_factory', operation='task_combined',
                ),
            ],
        )
        report = _make_report({
            'task_created_records': [
                {'task_id': '11', 'status': 'created', 'project_id': 'dark_factory'},
            ],
        })

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        assert report.stats['tasks_created'] == 2
        assert report.stats['tasks_combined'] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [('2', 2), (2.0, 2), (True, 0), (float('nan'), 0), (None, 0)],
        ids=['str', 'float', 'bool', 'nan', 'none'],
    )
    async def test_tasks_combined_self_report_is_coerced(self, raw, expected):
        stage = _make_stage()
        report = _make_report({'tasks_combined': raw})

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        assert report.stats['tasks_combined'] == expected
        assert type(report.stats['tasks_combined']) is int

    @pytest.mark.asyncio
    async def test_tasks_combined_is_always_present(self):
        stage = _make_stage()
        report = _make_report({})

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        assert report.stats['tasks_combined'] == 0
        assert 'tasks_combined_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_a_larger_tasks_combined_self_report_is_never_lowered(self):
        stage = _make_stage(journal_rows=[
            _journal_row(RUN_COMBINED, '42', project_id='dark_factory', operation='task_combined'),
        ])
        report = _make_report({'tasks_combined': 4})

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        assert report.stats['tasks_combined'] == 4
        assert 'tasks_combined_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_the_combined_stat_family_is_always_published(self):
        stage = _make_stage()
        report = _make_report({})

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        for suffix in ('valid', 'journaled', 'uncorroborated', 'unresolvable'):
            assert report.stats[f'tasks_combined_evidence_{suffix}'] == 0
        # Combined outcomes are recorded in task_created_records; no stat may
        # name a task_combined_records list that Stage 2 never emits.
        assert not [key for key in report.stats if key.startswith('task_combined_records')]

    @pytest.mark.asyncio
    async def test_tasks_combined_reported_carries_the_raw_value_on_repair(self):
        stage = _make_stage(journal_rows=[
            _journal_row(RUN_COMBINED, '42', project_id='dark_factory', operation='task_combined'),
        ])
        report = _make_report({'tasks_combined': '0'})

        await stage._apply_post_flight_guards(report, RUN_COMBINED)

        assert report.stats['tasks_combined'] == 1
        assert report.stats['tasks_combined_reported'] == '0'


# ── task 4018 (in 4873's scope): tasks_hints_updated parity ──────────────────

RUN_HINTS = '4018aaaa-5555-4000-8000-000000000007'


def _hinted_task(task_id: object, entities: list, queries: list) -> dict:
    return {
        'id': str(task_id),
        'title': 'A hinted task',
        'status': 'pending',
        'metadata': {'memory_hints': {'entities': entities, 'queries': queries}},
    }


class TestPostFlightHintsUpdatedGrounding:
    """``tasks_hints_updated`` is repaired from ``task_hints_updated_records``,
    each claim confirmed only when the re-read task's memory_hints contain
    every claimed entry (the SUPERSET rule)."""

    @pytest.mark.asyncio
    async def test_a_superset_confirmed_claim_repairs_the_counter(self):
        stub = _StubTaskmaster(results={
            ('7', '/repo/df'): _hinted_task(7, ['EntityA', 'Other'], ['how is A wired', 'q2']),
        })
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_hints_updated': 0,
            'task_hints_updated_records': [_hints_record()],
        })

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['tasks_hints_updated'] == 1
        assert report.stats['tasks_hints_updated_reported'] == 0
        assert report.stats['task_hints_updated_records_valid'] == 1

    @pytest.mark.asyncio
    async def test_hints_missing_a_claimed_query_do_not_confirm(self):
        stub = _StubTaskmaster(results={
            ('7', '/repo/df'): _hinted_task(7, ['EntityA'], []),
        })
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({'task_hints_updated_records': [_hints_record()]})

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['tasks_hints_updated'] == 0
        assert report.stats['task_hints_updated_records_uncorroborated'] == 1
        assert 'tasks_hints_updated_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_a_missing_task_does_not_confirm(self):
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        report = _make_report({'task_hints_updated_records': [_hints_record()]})

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['tasks_hints_updated'] == 0
        assert report.stats['task_hints_updated_records_uncorroborated'] == 1

    @pytest.mark.asyncio
    async def test_a_claim_in_an_unknown_project_is_unresolvable_with_no_lookup(self):
        stub = _StubTaskmaster()
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'task_hints_updated_records': [_hints_record(project_id='nowhere')],
        })

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['task_hints_updated_records_unresolvable'] == 1
        assert stub.calls == []

    @pytest.mark.asyncio
    async def test_the_lookup_goes_to_the_claims_own_project(self):
        stub = _StubTaskmaster(results={
            ('7', '/repo/reify'): _hinted_task(7, ['EntityA'], ['how is A wired']),
        })
        stage = _make_stage(
            known_projects={'dark_factory': '/repo/df', 'reify': '/repo/reify'},
            taskmaster=stub,
        )
        report = _make_report({
            'task_hints_updated_records': [_hints_record(project_id='reify')],
        })

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['tasks_hints_updated'] == 1
        assert stub.calls == [('7', '/repo/reify')]

    @pytest.mark.asyncio
    async def test_hints_have_no_journal_channel_and_the_counter_is_always_present(self):
        stage = _make_stage(
            known_projects=_KNOWN_PROJECTS,
            taskmaster=_StubTaskmaster(),
            journal_rows=[_journal_row(RUN_HINTS, '7', project_id='dark_factory')],
        )
        report = _make_report({})

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['tasks_hints_updated'] == 0
        assert report.stats['task_hints_updated_records_journaled'] == 0
        assert report.stats['task_hints_updated_records_valid'] == 0

    @pytest.mark.asyncio
    async def test_a_larger_self_report_is_never_lowered(self):
        stub = _StubTaskmaster(results={
            ('7', '/repo/df'): _hinted_task(7, ['EntityA'], ['how is A wired']),
        })
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'tasks_hints_updated': 3,
            'task_hints_updated_records': [_hints_record()],
        })

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['tasks_hints_updated'] == 3
        assert 'tasks_hints_updated_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_a_claim_naming_no_entries_does_not_count(self):
        stub = _StubTaskmaster(results={
            ('7', '/repo/df'): _hinted_task(7, ['EntityA'], ['how is A wired']),
        })
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=stub)
        report = _make_report({
            'task_hints_updated_records': [_hints_record(entities=[], queries=[])],
        })

        await stage._apply_post_flight_guards(report, RUN_HINTS)

        assert report.stats['tasks_hints_updated'] == 0
        assert report.stats['task_hints_updated_records_uncorroborated'] == 0
        assert stub.calls == []


class TestPostFlightFlagCountersAreNeverClamped:
    """Audit conclusion for the adjacent flag counters, as executable behaviour.

    Framework clamps once checked ``stage1_mem0_flags_processed`` against
    ``flag_deleted_records`` and ``stage1_analytical_findings_processed``
    against ``len(prior_reports[0].items_flagged)``, and ``prompts/stage2.py``
    never told the agent they were gone. Neither clamp has existed since
    tasks 2229/2230 (W5-mu) — ``_apply_post_flight_guards`` only
    ``setdefault``-normalizes both keys, and since task 4873 it no longer
    receives the prior reports at all. These pins keep that true (and are
    what makes the step-11 prompt correction verifiable in code rather than
    prose).
    """

    @pytest.mark.asyncio
    async def test_self_reported_flag_counters_survive_untouched(self):
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        report = _make_report({
            'stage1_analytical_findings_processed': 1,
            'stage1_mem0_flags_processed': 5,
            'flag_deleted_records': [
                {'action': 'flag_deleted', 'flag_id': f'flag-{i}'} for i in range(2)
            ],
        })

        await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['stage1_analytical_findings_processed'] == 1
        assert report.stats['stage1_mem0_flags_processed'] == 5

    @pytest.mark.asyncio
    async def test_absent_flag_counters_default_to_zero(self):
        stage = _make_stage(known_projects=_KNOWN_PROJECTS, taskmaster=_StubTaskmaster())
        report = _make_report({})

        await stage._apply_post_flight_guards(report, 'test-run-3051')

        assert report.stats['stage1_analytical_findings_processed'] == 0
        assert report.stats['stage1_mem0_flags_processed'] == 0
