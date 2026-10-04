"""Applying planned heals (task 6181, plans/write-triage-link-healing-prd.md H1 boundary rows).

Each case plans, then applies, against the in-process server in
``_link_heal_harness``, so what a heal did is observed through the real
server: the record's payload, grouped search and ``get_memory_by_id``, the
write journal and the ledger. Every heal re-reads the record live before its
one write, and re-reads it again after.
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
from fused_memory.maintenance.link_heal_executor import Escape, RunLimits, RunReport, run_apply
from fused_memory.maintenance.link_heal_ledger import (
    ActionRow,
    ActionState,
    LinkHealLedger,
    RunSource,
)
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


async def plan_then_apply(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    tmp_path: Path,
    *bases: LinkBasis,
    between: Callable[[], None] | None = None,
    limits: RunLimits = DEFAULT_LIMITS,
) -> RunReport:
    """Plan *bases*, run *between* (the world moving on), then apply."""
    await run_corpus_plan(harness, ledger, tmp_path / 'plan.json', bases, limits)
    if between is not None:
        between()
    return await run_apply(
        store=harness.store(),
        ledger=ledger,
        limits=limits,
        filer=RecordingFiler(),
        source=RunSource.CORPUS,
    )


def only_outcome(ledger: LinkHealLedger, report: RunReport) -> ActionRow:
    """The one heal *report*'s run executed, whatever became of it."""
    (row,) = [
        row
        for run in ledger.recent_runs(10)
        for row in ledger.run_actions(run.run_id)
        if row.executed_run_id == report.run_id
    ]
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
