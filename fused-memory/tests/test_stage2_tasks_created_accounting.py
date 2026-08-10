"""Wiring tests for the Task-Creation Accounting prompt contract (task 3046).

Stage 2 self-reports ``tasks_created`` as a bare LLM counter with no code-side
ground truth: ``submit_task``/``resolve_ticket`` are not journaled (see
``fused_memory.middleware.task_interceptor.TaskInterceptor._journal_around``,
which wraps only ``set_task_status``/``update_task``/``remove_tasks``/
``add_dependency``/``remove_dependency``), so ``derive_stage_stats`` cannot
recompute it and ``stats_verifier`` leaves it untouched (it is not in
``_COMPUTED_STAT_KEYS``). Run 507bc25b filed task 3045 via the Proactive Task
Sample / cross-project-routing surface yet Stage 2 self-reported
``tasks_created: 0`` — the increment mandate (``## Verifying Task
Operations``, ``prompts/stage2.py``) never declared the counter
path-agnostic, and (unlike the flag counters) had no action-record ground
truth analogous to ``flag_deleted_records``.

This module pins the WIRING of the fix's shared renderer,
``render_task_creation_accounting_section()`` (``recon_self_model.py``) —
exactly once per assembled Stage 2 prompt, on both
``build_stage2_system_prompt`` branches, never colliding with the
``'## Available Tools'`` injection sentinel or the ``mcp__recon-report__``
run_id drift guard (``test_recon_report_guidance_drift.py``) — and the
machine-readable contract tokens the Python side actually reads
(``tasks_created``, ``task_created_records``, the
``"action": "task_created"`` record shape). Per the
``test_recon_gate_closure_guidance.py`` house rule, prose wording beyond
those tokens is intentionally NOT pinned here.

It also pins the code-side layer: the pure ``_count_valid_task_created_records``
helper, and the deterministic upward-only repair
``TaskKnowledgeSync._apply_post_flight_guards`` performs when the
self-reported ``tasks_created`` undercounts the ``task_created_records``
ground truth — the exact shape of the run-507bc25b regression.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.recon_self_model import (
    render_task_creation_accounting_section,
)
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    TaskKnowledgeSync,
    _coerce_tasks_created_count,
    _count_valid_task_created_records,
)


class TestTaskCreationAccountingWiring:
    """render_task_creation_accounting_section() is wired into Stage 2 exactly once."""

    def test_embedded_verbatim_exactly_once_in_stage2_prompt(self):
        section = render_task_creation_accounting_section()

        # prompt.count(section) == 1 subsumes non-emptiness: str.count('')
        # over a non-empty STAGE2_SYSTEM_PROMPT is len(prompt) + 1, never 1.
        assert STAGE2_SYSTEM_PROMPT.count(section) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate '
            'render_task_creation_accounting_section() verbatim, exactly once.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        """Both runtime builder branches keep the section, exactly once.

        `autopilot_video` splices its contamination guardrail at the
        `'## Available Tools'` sentinel; that injection must not drop or
        duplicate this section.
        """
        section = render_task_creation_accounting_section()
        built = build_stage2_system_prompt(project_id)

        assert built.count(section) == 1, (
            f'build_stage2_system_prompt({project_id!r}) must carry '
            'render_task_creation_accounting_section() verbatim, exactly once.'
        )

    def test_assembled_prompt_carries_machine_readable_contract_tokens(self):
        """The Python side reads tasks_created / task_created_records / the
        "action": "task_created" record shape — pin those tokens, not prose.

        This also covers the '## Available Tools' sentinel and the
        mcp__recon-report__ run_id= guard by construction: if the rendered
        section carried either, `build_stage2_system_prompt('dark_factory')`
        would either raise (duplicated sentinel — see
        test_survives_build_stage2_system_prompt's autopilot_video branch)
        or would be caught separately by test_recon_report_guidance_drift.py,
        which scans this exact assembled prompt for run_id=-less call
        examples. No need to re-pin either as a standalone negative-substring
        check here (task-3046 amendment)."""
        built = build_stage2_system_prompt('dark_factory')

        assert 'tasks_created' in built
        assert 'task_created_records' in built
        assert '"action": "task_created"' in built


class TestCountValidTaskCreatedRecords:
    """Pins _count_valid_task_created_records (task 3046): a defensive,
    non-raising, deduped counter over stats['task_created_records'] — the
    code-side complement to the '## Verifying Task Operations' confirmation
    rule and the '## Task-Creation Accounting' action-record mandate.
    """

    # -- degenerate input: never raises, always 0 --------------------------

    @pytest.mark.parametrize(
        'records',
        [None, {}, 'x', 3, []],
        ids=['none', 'dict', 'str', 'int', 'empty-list'],
    )
    def test_degenerate_input_returns_zero(self, records):
        assert _count_valid_task_created_records(records) == 0

    # -- confirmation-status predicate ---------------------------------------

    def test_one_created_record_counts_one(self):
        records = [
            {
                'action': 'task_created',
                'task_id': '3045',
                'status': 'created',
                'project_id': 'dark_factory',
            },
        ]
        assert _count_valid_task_created_records(records) == 1

    def test_combined_status_counts_as_success(self):
        """`combined` is a success per `## Creating Tasks` — a candidate
        merged into an existing task still returns a task_id."""
        records = [
            {
                'action': 'task_created',
                'task_id': '3045',
                'status': 'combined',
                'project_id': 'dark_factory',
            },
        ]
        assert _count_valid_task_created_records(records) == 1

    def test_failed_status_never_counts_even_with_task_id(self):
        """`failed` is never counted toward tasks_created regardless of
        whether a task_id is present — mirrors the '## Verifying Task
        Operations' rule this helper encodes in Python."""
        records = [
            {
                'action': 'task_created',
                'task_id': '3045',
                'status': 'failed',
                'project_id': 'dark_factory',
            },
        ]
        assert _count_valid_task_created_records(records) == 0

    @pytest.mark.parametrize(
        'record',
        [
            {'status': 'created', 'project_id': 'dark_factory'},
            {'status': 'created', 'project_id': 'dark_factory', 'task_id': None},
            {'status': 'created', 'project_id': 'dark_factory', 'task_id': ''},
        ],
        ids=['missing-task_id', 'none-task_id', 'empty-task_id'],
    )
    def test_missing_or_empty_task_id_does_not_count(self, record):
        assert _count_valid_task_created_records([record]) == 0

    @pytest.mark.parametrize(
        'record',
        [
            {'task_id': '3045', 'status': 'pending', 'project_id': 'dark_factory'},
            {'task_id': '3045', 'project_id': 'dark_factory'},
        ],
        ids=['unknown-status', 'missing-status'],
    )
    def test_unknown_or_missing_status_does_not_count(self, record):
        """Only created/combined count — the helper does not guess at an
        absent or unrecognized status."""
        assert _count_valid_task_created_records([record]) == 0

    # -- malformed entries are skipped, never raise --------------------------

    def test_non_dict_entries_interleaved_with_one_valid_record(self):
        records = [
            None,
            'x',
            ['a'],
            {
                'action': 'task_created',
                'task_id': '3045',
                'status': 'created',
                'project_id': 'dark_factory',
            },
        ]
        assert _count_valid_task_created_records(records) == 1

    # -- dedup: keyed on (project_id, str(task_id)) --------------------------

    def test_dedups_same_project_and_task_id(self):
        records = [
            {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            {'task_id': '3045', 'status': 'combined', 'project_id': 'dark_factory'},
        ]
        assert _count_valid_task_created_records(records) == 1

    def test_same_task_id_under_different_projects_counts_twice(self):
        """Cross-project routing files genuinely distinct tasks — Taskmaster
        ids are per-project, so the same numeric id under two different
        project_ids is two distinct tasks, not a duplicate."""
        records = [
            {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            {'task_id': '3045', 'status': 'created', 'project_id': 'autopilot_video'},
        ]
        assert _count_valid_task_created_records(records) == 2

    def test_dedups_int_and_str_task_id_forms(self):
        """task_id normalises to str before deduping — the repo's existing
        str(task_id) exact-form convention."""
        records = [
            {'task_id': 3045, 'status': 'created', 'project_id': 'dark_factory'},
            {'task_id': '3045', 'status': 'combined', 'project_id': 'dark_factory'},
        ]
        assert _count_valid_task_created_records(records) == 1

    # -- project_id normalisation (task-3046 amendment) -----------------------

    def test_dedups_project_id_with_surrounding_whitespace(self):
        """project_id is normalised via str(...).strip(), same as task_id —
        ' dark_factory' and 'dark_factory' must collapse to one key, not
        inflate the count by masquerading as a second distinct project."""
        records = [
            {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            {'task_id': '3045', 'status': 'combined', 'project_id': ' dark_factory '},
        ]
        assert _count_valid_task_created_records(records) == 1

    def test_missing_project_id_collapses_onto_default_project_id(self):
        """A record with no project_id falls back to default_project_id (the
        caller's own self.project_id) rather than keying as a distinct
        `None` project — so an omitted field can't masquerade as a second,
        distinct cross-project filing of the same task."""
        records = [
            {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            {'task_id': '3045', 'status': 'combined'},  # project_id omitted
        ]
        assert _count_valid_task_created_records(
            records, default_project_id='dark_factory'
        ) == 1

    def test_missing_project_id_without_default_keys_as_none(self):
        """With no default_project_id supplied (the pre-amendment call
        shape), a record with no project_id still keys as None — same
        behaviour as before this amendment when the caller opts out of the
        default-collapse."""
        records = [
            {'task_id': '3045', 'status': 'created'},
            {'task_id': '3045', 'status': 'combined'},
        ]
        assert _count_valid_task_created_records(records) == 1

    def test_blank_project_id_after_strip_collapses_onto_default(self):
        """A whitespace-only project_id is treated as omitted, not as a
        literal '' project — it also falls back to default_project_id."""
        records = [
            {'task_id': '3045', 'status': 'created', 'project_id': 'dark_factory'},
            {'task_id': '3045', 'status': 'combined', 'project_id': '   '},
        ]
        assert _count_valid_task_created_records(
            records, default_project_id='dark_factory'
        ) == 1


class TestCoerceTasksCreatedCount:
    """Pins _coerce_tasks_created_count (task-3046 amendment): best-effort int
    coercion for a self-reported tasks_created value, so a non-int self-report
    (plausible free-form LLM JSON) doesn't collapse to 0 and look like an
    undercount relative to a smaller record-derived observed count — which
    would silently move a legitimately larger self-report DOWN, breaking the
    upward-only repair contract (task 2230 / W5-mu)."""

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            (0, 0),
            (3, 3),
            ('0', 0),
            ('3', 3),
            (' 3 ', 3),
            (3.0, 3),
            ('3.0', 3),
        ],
        ids=['int-0', 'int-3', 'str-0', 'str-3', 'str-3-whitespace', 'float-3', 'str-float-3'],
    )
    def test_coerces_legitimate_numeric_forms(self, value, expected):
        assert _coerce_tasks_created_count(value) == expected

    @pytest.mark.parametrize(
        'value',
        [None, True, False, 'not-a-number', [], {}],
        ids=['none', 'true', 'false', 'non-numeric-str', 'list', 'dict'],
    )
    def test_uncoercible_or_bool_falls_back_to_zero(self, value):
        """A bool is REJECTED even though isinstance(True, int) holds in
        Python — never trusted as a legitimate self-reported count."""
        assert _coerce_tasks_created_count(value) == 0


# ── Task-3046 step-5: TaskKnowledgeSync._apply_post_flight_guards repair ────


def _mock_deps() -> dict:
    """Kwargs for constructing a TaskKnowledgeSync with mocked deps (task 3046).

    Replicates the ~12-line construction from
    ``tests/test_stages.py::stage2_guard_mock_deps`` (10083-10098) rather than
    cross-importing that module-private fixture — the same choice
    ``test_recon_gate_closure_guidance.py::_make_consolidator`` documents and
    makes for ``MemoryConsolidator``.
    """
    config = ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test')
    write_journal_mock = MagicMock()
    write_journal_mock.get_ops_by_causation = AsyncMock(return_value=[])
    journal_mock = MagicMock()
    journal_mock.write_journal = write_journal_mock
    journal_mock.record_run_session = AsyncMock()
    journal_mock.clear_run_session = AsyncMock()
    return {
        'memory_service': AsyncMock(),
        'taskmaster': AsyncMock(),
        'journal': journal_mock,
        'config': config,
        'scope': ProjectScope(ProjectId('test_project'), ProjectRoot('/tmp/test')),
    }


def _make_stage() -> TaskKnowledgeSync:
    return TaskKnowledgeSync(StageId.task_knowledge_sync, **_mock_deps())


def _make_report(stats: dict) -> StageReport:
    now = datetime.now(tz=UTC)
    return StageReport(
        stage=StageId.task_knowledge_sync,
        started_at=now,
        completed_at=now,
        stats=stats,
    )


_REPAIR_LOGGER = 'fused_memory.reconciliation.stages.task_knowledge_sync'


class TestPostFlightTasksCreatedRepair:
    """TaskKnowledgeSync._apply_post_flight_guards' tasks_created repair (task 3046).

    Exercises the post-flight guard directly (not via ``stage.run()``): with
    ``flag_deleted_records`` absent from stats,
    ``_acknowledge_resolved_stage1_markers`` short-circuits to 0 without
    touching the mocked memory service, isolating these assertions to the
    tasks_created normalization + upward-only repair added by step-6.
    """

    @pytest.mark.asyncio
    async def test_normalization_on_empty_stats(self):
        """No counter, no records: both new keys default to 0, no repair key."""
        stage = _make_stage()
        report = _make_report({})

        await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert 'tasks_created_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_run_507bc25b_regression_undercount_is_repaired(self, caplog):
        """The exact run-507bc25b shape: tasks_created=0 self-reported while
        one task_created_records entry confirms task 3045 was filed via the
        proactive-sample surface — repaired upward to 1."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                    'source_path': 'proactive_sample',
                },
            ],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == 1
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 1

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'tasks_created=0' in message
        assert 'to 1' in message

    @pytest.mark.asyncio
    async def test_missing_counter_key_derives_from_records(self):
        """tasks_created absent entirely; two distinct confirmed records ⇒ 2."""
        stage = _make_stage()
        report = _make_report({
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
                {
                    'action': 'task_created',
                    'task_id': '3046',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == 2
        assert report.stats['tasks_created_reported'] == 0
        assert report.stats['task_created_records_valid'] == 2

    @pytest.mark.asyncio
    async def test_agreement_leaves_counter_untouched_no_repair_key(self, caplog):
        """Self-report already matches the record-derived count: no repair,
        no spurious tasks_created_reported key, no WARNING."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 2,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
                {
                    'action': 'task_created',
                    'task_id': '3046',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == 2
        assert 'tasks_created_reported' not in report.stats
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_no_downward_clamp_on_over_report(self, caplog):
        """Mirrors test_stages.py::TestPostFlightGuardsNoFlagCounterClamp
        (task 2230): an over-reported tasks_created survives untouched even
        though ground truth is lower. task_created_records_valid still
        publishes the lower count so the divergence stays observable, and no
        WARNING fires (this guard warns only on an UNDER-count repair)."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 3,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        with caplog.at_level(logging.WARNING, logger=_REPAIR_LOGGER):
            await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == 3
        assert report.stats['task_created_records_valid'] == 1
        assert 'tasks_created_reported' not in report.stats
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_non_list_records_are_inert(self):
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': 'not-a-list',
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert 'tasks_created_reported' not in report.stats

    @pytest.mark.asyncio
    async def test_non_dict_and_failed_status_entries_are_inert(self):
        stage = _make_stage()
        report = _make_report({
            'tasks_created': 0,
            'task_created_records': [
                None,
                'x',
                ['a'],
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'failed',
                    'project_id': 'dark_factory',
                },
            ],
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == 0
        assert report.stats['task_created_records_valid'] == 0
        assert 'tasks_created_reported' not in report.stats

    # -- non-int self-reports never repair downward (task-3046 amendment) ---

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('raw_tasks_created', 'expected', 'repaired'),
        [
            ('0', 1, True),    # numeric str below ground truth -> repaired up
            ('3', 3, False),   # numeric str at/above ground truth -> normalized only
            (3.0, 3, False),   # float at/above ground truth -> normalized only
            (True, 1, True),   # bool rejected outright -> treated as unreported -> repaired up
            (None, 1, True),   # explicit None value -> treated as unreported -> repaired up
        ],
        ids=[
            'str-zero-undercounts',
            'str-three-agrees',
            'float-three-agrees',
            'bool-rejected',
            'none-value',
        ],
    )
    async def test_non_int_self_report_never_repaired_downward(
        self, raw_tasks_created, expected, repaired,
    ):
        """A non-int self-report must coerce to its real numeric value
        before the upward-only comparison, never collapse to 0 and get
        spuriously "repaired" DOWN from a legitimately larger self-report
        (review suggestion #1). The post-guard value is always a plain
        ``int`` (never a residual ``str``/``float``/``bool``)."""
        stage = _make_stage()
        report = _make_report({
            'tasks_created': raw_tasks_created,
            'task_created_records': [
                {
                    'action': 'task_created',
                    'task_id': '3045',
                    'status': 'created',
                    'project_id': 'dark_factory',
                },
            ],
        })

        await stage._apply_post_flight_guards(report, [], 'test-run-3046')

        assert report.stats['tasks_created'] == expected
        assert isinstance(report.stats['tasks_created'], int)
        assert not isinstance(report.stats['tasks_created'], bool)
        assert ('tasks_created_reported' in report.stats) is repaired
        if repaired:
            assert report.stats['tasks_created_reported'] == raw_tasks_created
