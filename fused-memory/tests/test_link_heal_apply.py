"""Applying planned heals (task 6181, plans/write-triage-link-healing-prd.md H1 boundary rows).

Each case plans, then applies, against the in-process server in
``_link_heal_harness``, so what a heal did is observed through the real
server: the record's payload, grouped search and ``get_memory_by_id``, the
write journal and the ledger. Every heal re-reads the record live before its
one write, and re-reads it again after.

The second section covers what bounds a run: the per-run cap, the backlog and
write-failure escapes, and an operator-approved plan.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio
from _link_heal_harness import (
    ALL_LINK_HEAL_PREFIXES,
    CHILD,
    CHILD_TEXT,
    DEFAULT_LIMITS,
    DF,
    FORBIDDEN_ROUTES,
    PARENT,
    PARENT_TEXT,
    WITHOUT_LINK_HEAL_PREFIX,
    LinkHealHarness,
    build_harness,
    corpus_basis,
    run_corpus_plan,
)
from orchestrator.agents.memory_recall import render_memory_results

from fused_memory.maintenance.link_heal import LINK_KEYS, BasisSource, LinkBasis
from fused_memory.maintenance.link_heal_executor import (
    BACKLOG_ANCHOR,
    WRITE_FAILURE_ANCHOR,
    ApprovalMismatch,
    Escape,
    EscapeFiler,
    FoldedEscapeFiler,
    RunLimits,
    RunReport,
    plan_sha256,
    render_plan_document,
    run_apply,
)
from fused_memory.maintenance.link_heal_ledger import (
    ActionRow,
    ActionState,
    LinkHealLedger,
    RunSource,
)
from fused_memory.middleware import _folded_escalation
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)

OTHER_PARENT = '33333333-3333-4333-8333-333333333333'
GRANDCHILD = '55555555-5555-4555-8555-555555555555'


def assert_store_invariants(harness: LinkHealHarness) -> None:
    """No heal combines a patch and a delete, sends content, or stores a null link key."""
    for route in FORBIDDEN_ROUTES:
        assert harness.mem0.writes[route] == 0, route
    for (project_id, memory_id), payload in harness.mem0.points.items():
        for key in LINK_KEYS:
            assert key not in payload or payload[key] is not None, (project_id, memory_id, key)


async def _harness(mock_config, tmp_path: Path, prefixes: list[str]):
    built = await build_harness(mock_config, tmp_path, metadata_patch_prefixes=prefixes)
    yield built
    await built.journal.close()
    assert_store_invariants(built)


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    async for built in _harness(mock_config, tmp_path, ALL_LINK_HEAL_PREFIXES):
        yield built


@pytest_asyncio.fixture
async def unadmitted(mock_config, tmp_path):
    """A server whose metadata allowlist does not admit ``link-heal-``."""
    async for built in _harness(mock_config, tmp_path, WITHOUT_LINK_HEAL_PREFIX):
        yield built


@pytest.fixture
def ledger(tmp_path):
    opened = LinkHealLedger(tmp_path / 'link_heal.db')
    yield opened
    opened.close()


class RecordingFiler:
    """An ``EscapeFiler`` that files nowhere and remembers every escape."""

    def __init__(self) -> None:
        self.escapes: list[Escape] = []

    def __call__(self, escape: Escape) -> str | None:
        self.escapes.append(escape)
        return f'esc-recorded-{len(self.escapes)}'


async def apply_pending(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    *,
    limits: RunLimits = DEFAULT_LIMITS,
    filer: EscapeFiler | None = None,
    approved_plan_sha256: str | None = None,
) -> RunReport:
    return await run_apply(
        store=harness.store(),
        ledger=ledger,
        limits=limits,
        filer=filer or RecordingFiler(),
        source=RunSource.CORPUS,
        approved_plan_sha256=approved_plan_sha256,
    )


async def plan_then_apply(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    tmp_path: Path,
    *bases: LinkBasis,
    between: Callable[[], None] | None = None,
    limits: RunLimits = DEFAULT_LIMITS,
    filer: EscapeFiler | None = None,
) -> RunReport:
    """Plan *bases*, run *between* (the world moving on), then apply."""
    await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', bases, limits)
    if between is not None:
        between()
    return await apply_pending(harness, ledger, limits=limits, filer=filer)


def executed_rows(ledger: LinkHealLedger, report: RunReport) -> list[ActionRow]:
    """Every heal *report*'s run executed, whatever became of it, oldest first."""
    rows = [
        row
        for run in ledger.recent_runs(10)
        for row in ledger.run_actions(run.run_id)
        if row.executed_run_id == report.run_id
    ]
    return sorted(rows, key=lambda row: row.action_id)


def only_outcome(ledger: LinkHealLedger, report: RunReport) -> ActionRow:
    (row,) = executed_rows(ledger, report)
    return row


class TestDetach:
    @pytest.mark.asyncio
    async def test_a_misfiled_amendment_is_detached_and_surfaces_on_its_own(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()

        report = await plan_then_apply(harness, ledger, tmp_path, corpus_basis('RELATED'))

        payload = harness.mem0.payload(DF, CHILD)
        assert PARENT_ID_KEY not in payload
        assert 'kind' not in payload
        search = await harness.call('search', query='the child note', project_id=DF)
        assert [hit['id'] for hit in search['results']] == [CHILD]
        assert 'parent_unresolved' not in search['results'][0]
        parent = await harness.call('get_memory_by_id', project_id=DF, memory_id=PARENT)
        assert 'grouped' not in parent
        assert report.counts.applied == 1

    @pytest.mark.asyncio
    async def test_the_detach_is_ledgered_applied_by_the_apply_run(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()

        report = await plan_then_apply(harness, ledger, tmp_path, corpus_basis('RELATED'))

        (row,) = ledger.applied_actions(report.run_id)
        assert row.executed_run_id == report.run_id
        assert row.state is ActionState.APPLIED
        assert ledger.pending_actions(RunSource.CORPUS) == []

    @pytest.mark.asyncio
    async def test_the_detach_is_one_attributed_delete_only_journal_row(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()

        report = await plan_then_apply(harness, ledger, tmp_path, corpus_basis('RELATED'))

        (row,) = harness.journal_rows()
        assert row['agent_id'] == f'link-heal-{report.run_id[:8]}'
        assert row['causation_id'] == report.run_id
        reason = row['params']['reason']
        assert reason.startswith('link-heal r=')
        assert len(reason) < 200
        assert row['params']['metadata_delete_keys'] == [PARENT_ID_KEY, 'kind']
        assert 'metadata_patch' not in row['params']

    @pytest.mark.asyncio
    async def test_a_misfiled_extension_half_link_keeps_its_kind(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind='extension')

        await plan_then_apply(harness, ledger, tmp_path, corpus_basis('RELATED'))

        payload = harness.mem0.payload(DF, CHILD)
        assert PARENT_ID_KEY not in payload
        assert payload['kind'] == 'extension'

    @pytest.mark.asyncio
    async def test_a_dangling_link_is_detached_on_a_deterministic_basis(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(parent_text=None)

        report = await plan_then_apply(harness, ledger, tmp_path)

        (row,) = ledger.applied_actions(report.run_id)
        assert row.planned.basis_source is BasisSource.DETERMINISTIC
        assert PARENT_ID_KEY not in harness.mem0.payload(DF, CHILD)


class TestRelabelFlagComplete:
    @pytest.mark.asyncio
    async def test_an_extends_sighting_is_relabelled_into_the_parents_amendments(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=SIGHTING_KIND)

        await plan_then_apply(harness, ledger, tmp_path, corpus_basis('EXTENDS'))

        assert harness.mem0.payload(DF, CHILD)['kind'] == AMENDMENT_KIND
        parent = await harness.call('get_memory_by_id', project_id=DF, memory_id=PARENT)
        assert [entry['id'] for entry in parent['grouped']['amendments']] == [CHILD]

    @pytest.mark.asyncio
    async def test_a_relabelled_amendment_renders_nested_under_its_parent(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=SIGHTING_KIND)
        await plan_then_apply(harness, ledger, tmp_path, corpus_basis('EXTENDS'))

        search = await harness.call('search', query='link healing', project_id=DF)
        lines = render_memory_results(search['results']).splitlines()

        parent_line = next(i for i, line in enumerate(lines) if PARENT_TEXT in line)
        assert lines[parent_line].startswith('- ')
        assert lines[parent_line + 1].startswith('  - ')
        assert CHILD_TEXT in lines[parent_line + 1]

    @pytest.mark.asyncio
    async def test_a_corrects_amendment_is_flagged_and_never_suppressed(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()

        await plan_then_apply(harness, ledger, tmp_path, corpus_basis('CORRECTS'))

        payload = harness.mem0.payload(DF, CHILD)
        assert payload[CONTESTED_METADATA_KEY] is True
        assert payload['kind'] == AMENDMENT_KIND
        search = await harness.call('search', query='link healing', project_id=DF)
        assert CHILD in [hit['id'] for hit in search['results']]

    @pytest.mark.asyncio
    async def test_a_no_kind_same_half_link_is_completed_to_a_sighting(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=None)
        before = await harness.call('get_memory_by_id', project_id=DF, memory_id=PARENT)

        await plan_then_apply(harness, ledger, tmp_path, corpus_basis('SAME'))

        assert harness.mem0.payload(DF, CHILD)['kind'] == SIGHTING_KIND
        after = await harness.call('get_memory_by_id', project_id=DF, memory_id=PARENT)
        assert after['grouped']['sighting_count'] == before['grouped']['sighting_count'] + 1


def _edit_child_text(harness: LinkHealHarness) -> Callable[[], None]:
    def edit() -> None:
        harness.mem0.payload(DF, CHILD)['data'] = 'the child note, edited after the plan'

    return edit


class TestCorroborationSkipsStaleHeals:
    @pytest.mark.asyncio
    async def test_a_child_edited_after_the_plan_is_skipped_stale(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()

        report = await plan_then_apply(
            harness, ledger, tmp_path, corpus_basis('RELATED'),
            between=_edit_child_text(harness),
        )

        row = only_outcome(ledger, report)
        assert row.state is ActionState.SKIPPED_STALE
        assert row.detail == {'stale_field': 'child_sha256'}
        assert harness.mem0.write_count == 0
        assert report.counts.skipped_stale == 1

    @pytest.mark.asyncio
    async def test_a_child_re_parented_after_the_plan_is_skipped_stale(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()
        harness.seed(DF, OTHER_PARENT, 'a parent it was folded under since')

        def refold() -> None:
            harness.mem0.payload(DF, CHILD)[PARENT_ID_KEY] = OTHER_PARENT

        report = await plan_then_apply(
            harness, ledger, tmp_path, corpus_basis('RELATED'), between=refold,
        )

        row = only_outcome(ledger, report)
        assert row.state is ActionState.SKIPPED_STALE
        assert row.detail == {'stale_field': PARENT_ID_KEY}
        assert harness.mem0.payload(DF, CHILD)[PARENT_ID_KEY] == OTHER_PARENT

    @pytest.mark.asyncio
    async def test_a_child_gaining_a_child_before_completion_is_skipped_stale(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link(kind=None)

        def adopt() -> None:
            harness.seed(
                DF, GRANDCHILD, 'a note filed under the child since',
                **{PARENT_ID_KEY: CHILD, 'kind': SIGHTING_KIND},
            )

        report = await plan_then_apply(
            harness, ledger, tmp_path, corpus_basis('SAME'), between=adopt,
        )

        row = only_outcome(ledger, report)
        assert row.state is ActionState.SKIPPED_STALE
        assert row.detail == {'stale_field': 'children'}
        assert 'kind' not in harness.mem0.payload(DF, CHILD)


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_parent_read_timeout_fails_the_heal_and_never_detaches(
        self, harness, ledger, tmp_path,
    ):
        harness.seed_link()

        report = await plan_then_apply(
            harness, ledger, tmp_path, corpus_basis('RELATED'),
            between=lambda: harness.mem0.read_failures.add(PARENT),
        )

        row = only_outcome(ledger, report)
        assert row.state is ActionState.FAILED
        assert row.detail is not None
        assert row.detail['failure'] == 'read'
        assert row.detail['memory_id'] == PARENT
        assert row.detail['tool'] == 'get_memory_by_id'
        assert harness.mem0.payload(DF, CHILD)[PARENT_ID_KEY] == PARENT
        assert harness.mem0.write_count == 0
        assert report.counts.failed == 1
        assert report.counts.complete is False

    @pytest.mark.asyncio
    async def test_an_error_type_reply_is_a_failed_write(self, unadmitted, ledger, tmp_path):
        unadmitted.seed_link(kind=SIGHTING_KIND)

        report = await plan_then_apply(unadmitted, ledger, tmp_path, corpus_basis('EXTENDS'))

        row = only_outcome(ledger, report)
        assert row.state is ActionState.FAILED
        assert row.detail is not None
        assert row.detail['failure'] == 'write'
        assert row.detail['error_type'] == 'Mem0UpdateNotAuthorized'
        assert unadmitted.mem0.payload(DF, CHILD)['kind'] == SIGHTING_KIND
        assert unadmitted.journal_rows() == []
        assert report.counts.failed == 1


# ---------------------------------------------------------------------------
# Caps, escapes and approval.
# ---------------------------------------------------------------------------


def _sighting_ids(index: int) -> tuple[str, str]:
    return (f'{index:08x}-c0c0-4c0c-8c0c-{index:012x}', f'{index:08x}-a0a0-4a0a-8a0a-{index:012x}')


def seed_sightings(harness: LinkHealHarness, count: int) -> tuple[LinkBasis, ...]:
    """*count* EXTENDS-rated sightings, each under its own parent: *count* relabels."""
    bases = []
    for index in range(count):
        child, parent = _sighting_ids(index)
        child_text, parent_text = f'sighting {index}', f'parent {index}'
        harness.seed_link(
            kind=SIGHTING_KIND, child=child, parent=parent,
            child_text=child_text, parent_text=parent_text,
        )
        bases.append(corpus_basis(
            'EXTENDS', child=child, parent=parent,
            child_text=child_text, parent_text=parent_text, key=f'H{index:03d}',
        ))
    return tuple(bases)


def _ids(rows: list[ActionRow]) -> list[int]:
    return [row.action_id for row in rows]


class TestTheCapDrainsOldestFirst:
    @pytest.mark.asyncio
    async def test_a_capped_run_applies_the_oldest_and_leaves_the_rest_pending(
        self, harness, ledger, tmp_path,
    ):
        await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', seed_sightings(harness, 40))
        planned = _ids(ledger.pending_actions(RunSource.CORPUS))

        report = await apply_pending(harness, ledger)

        assert _ids(ledger.applied_actions(report.run_id)) == planned[:25]
        rest = ledger.pending_actions(RunSource.CORPUS)
        assert _ids(rest) == planned[25:]
        assert {row.state for row in rest} == {ActionState.SKIPPED_CAP}
        assert (report.counts.applied, report.counts.skipped_cap) == (25, 15)
        assert report.counts.caps_bit == ('max_actions_per_run',)
        assert report.counts.complete is False

    @pytest.mark.asyncio
    async def test_the_next_run_drains_exactly_the_cap_skipped_rest(
        self, harness, ledger, tmp_path,
    ):
        await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', seed_sightings(harness, 40))
        planned = _ids(ledger.pending_actions(RunSource.CORPUS))
        await apply_pending(harness, ledger)

        report = await apply_pending(harness, ledger)

        assert _ids(ledger.applied_actions(report.run_id)) == planned[25:]
        assert report.counts.skipped_cap == 0
        assert report.counts.caps_bit == ()
        assert report.counts.complete is True


BACKLOG_LIMITS = RunLimits(max_actions_per_run=25, backlog_multiplier=1, write_failure_streak=3)


class TestTheBacklogEscape:
    @pytest.mark.asyncio
    async def test_a_backlog_files_one_escape_and_the_run_keeps_draining(
        self, harness, ledger, tmp_path,
    ):
        filer = RecordingFiler()

        report = await plan_then_apply(
            harness, ledger, tmp_path, *seed_sightings(harness, 40),
            limits=BACKLOG_LIMITS, filer=filer,
        )

        (escape,) = filer.escapes
        assert escape.anchor == BACKLOG_ANCHOR
        assert report.counts.applied == 25
        assert report.counts.escaped == (
            {'anchor': BACKLOG_ANCHOR, 'escalation_id': 'esc-recorded-1'},
        )

    @pytest.mark.skipif(
        not _folded_escalation.HAS_ESCALATION,
        reason='escalation package unavailable (minimal env)',
    )
    @pytest.mark.asyncio
    async def test_the_real_filer_files_once_and_a_second_capped_run_folds_in(
        self, harness, ledger, tmp_path,
    ):
        from escalation.queue import EscalationQueue

        filer = FoldedEscapeFiler(project_root=str(tmp_path))
        bases = seed_sightings(harness, 60)

        first = await plan_then_apply(
            harness, ledger, tmp_path, *bases, limits=BACKLOG_LIMITS, filer=filer,
        )
        second = await apply_pending(harness, ledger, limits=BACKLOG_LIMITS, filer=filer)

        queue = EscalationQueue(tmp_path / 'data' / 'escalations')
        (record,) = queue.get_by_task(BACKLOG_ANCHOR, status='pending')
        assert 'projects: ["dark_factory"]' in record.detail
        assert 'pending: 60' in record.detail
        assert first.counts.escaped == ({'anchor': BACKLOG_ANCHOR, 'escalation_id': record.id},)
        assert second.counts.escaped == ({'anchor': BACKLOG_ANCHOR, 'escalation_id': record.id},)


STREAK_LIMITS = RunLimits(max_actions_per_run=25, backlog_multiplier=5, write_failure_streak=3)


class TestTheWriteFailureStreak:
    @pytest.mark.asyncio
    async def test_an_unadmitted_prefix_stops_the_run_after_the_streak(
        self, unadmitted, ledger, tmp_path,
    ):
        filer = RecordingFiler()

        report = await plan_then_apply(
            unadmitted, ledger, tmp_path, *seed_sightings(unadmitted, 5),
            limits=STREAK_LIMITS, filer=filer,
        )

        counts = report.counts
        assert counts.stopped_by == 'write_failure_streak'
        assert (counts.failed, counts.not_attempted) == (3, 2)
        assert counts.complete is False
        rest = ledger.pending_actions(RunSource.CORPUS)
        assert len(rest) == 2
        assert {row.state for row in rest} == {ActionState.PLANNED}
        (escape,) = filer.escapes
        assert escape.anchor == WRITE_FAILURE_ANCHOR
        assert counts.escaped == (
            {'anchor': WRITE_FAILURE_ANCHOR, 'escalation_id': 'esc-recorded-1'},
        )

    @pytest.mark.asyncio
    async def test_the_first_failure_names_the_refusal(self, unadmitted, ledger, tmp_path):
        report = await plan_then_apply(
            unadmitted, ledger, tmp_path, *seed_sightings(unadmitted, 5), limits=STREAK_LIMITS,
        )

        first = executed_rows(ledger, report)[0]
        assert first.state is ActionState.FAILED
        assert first.detail is not None
        assert first.detail['error_type'] == 'Mem0UpdateNotAuthorized'

    def _sabotage(self, harness: LinkHealHarness, pattern: str) -> Callable[[], None]:
        """Per sighting, in plan order: f fails its parent read, s goes stale, o is left ok."""

        def between() -> None:
            for index, mark in enumerate(pattern):
                child, parent = _sighting_ids(index)
                if mark == 'f':
                    harness.mem0.read_failures.add(parent)
                elif mark == 's':
                    harness.mem0.payload(DF, child)['data'] = f'sighting {index}, edited'

        return between

    async def _run(self, harness, ledger, tmp_path, pattern: str) -> RunReport:
        return await plan_then_apply(
            harness, ledger, tmp_path, *seed_sightings(harness, len(pattern)),
            between=self._sabotage(harness, pattern),
            limits=RunLimits(max_actions_per_run=25, backlog_multiplier=5, write_failure_streak=2),
        )

    @pytest.mark.asyncio
    async def test_a_stale_skip_does_not_break_the_streak(self, harness, ledger, tmp_path):
        counts = (await self._run(harness, ledger, tmp_path, 'fsfo')).counts

        assert counts.stopped_by == 'write_failure_streak'
        assert (counts.failed, counts.skipped_stale, counts.not_attempted) == (2, 1, 1)

    @pytest.mark.asyncio
    async def test_a_stale_skip_does_not_extend_the_streak(self, harness, ledger, tmp_path):
        counts = (await self._run(harness, ledger, tmp_path, 'fso')).counts

        assert counts.stopped_by is None
        assert (counts.failed, counts.skipped_stale, counts.applied) == (1, 1, 1)

    @pytest.mark.asyncio
    async def test_an_applied_heal_resets_the_streak(self, harness, ledger, tmp_path):
        counts = (await self._run(harness, ledger, tmp_path, 'fofo')).counts

        assert counts.stopped_by is None
        assert (counts.failed, counts.applied, counts.not_attempted) == (2, 2, 0)


class TestAnApprovedPlan:
    @pytest.mark.asyncio
    async def test_an_approved_plan_lifts_the_cap_and_skips_the_backlog_escape(
        self, harness, ledger, tmp_path,
    ):
        await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', seed_sightings(harness, 40))
        sha = plan_sha256(render_plan_document(ledger.pending_actions(RunSource.CORPUS)))
        filer = RecordingFiler()

        report = await apply_pending(
            harness, ledger, limits=BACKLOG_LIMITS, filer=filer, approved_plan_sha256=sha,
        )

        assert report.counts.applied == 40
        assert report.counts.skipped_cap == 0
        assert report.counts.approved_plan_sha256 == sha
        assert filer.escapes == []
        assert report.counts.escaped == ()

    @pytest.mark.asyncio
    async def test_a_wrong_sha_is_refused_before_any_run_row(self, harness, ledger, tmp_path):
        await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', seed_sightings(harness, 3))
        sha = plan_sha256(render_plan_document(ledger.pending_actions(RunSource.CORPUS)))
        runs_before = ledger.recent_runs(10)

        with pytest.raises(ApprovalMismatch) as excinfo:
            await apply_pending(harness, ledger, approved_plan_sha256='0' * 64)

        assert '0' * 64 in str(excinfo.value)
        assert sha in str(excinfo.value)
        assert ledger.recent_runs(10) == runs_before
        assert harness.mem0.write_count == 0
