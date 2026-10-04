"""Planning heals over a live store (task 6181, plans/write-triage-link-healing-prd.md H1).

Every case runs ``build_plan`` against the in-process server in
``_link_heal_harness``: the census enumerates the linked ids, and every value
the table reads comes back through the server's own tools.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from _link_heal_harness import ALL_LINK_HEAL_PREFIXES, LinkHealHarness, build_harness

from fused_memory.maintenance.link_heal import (
    BasisSource,
    HealAction,
    LinkBasis,
    LinkImage,
    Plan,
    RunCounts,
    Verdict,
    build_plan,
)
from fused_memory.maintenance.link_heal_store import text_sha256
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)

DF = 'dark_factory'
REIFY = 'reify'
PROJECTS = (DF, REIFY)

CHILD = '11111111-1111-4111-8111-111111111111'
PARENT = '22222222-2222-4222-8222-222222222222'
OTHER_PARENT = '33333333-3333-4333-8333-333333333333'
GRANDPARENT = '44444444-4444-4444-8444-444444444444'
GRANDCHILD = '55555555-5555-4555-8555-555555555555'
SECOND_CHILD = '66666666-6666-4666-8666-666666666666'

CHILD_TEXT = 'the child note about link healing'
PARENT_TEXT = 'the parent note about link healing'


@pytest_asyncio.fixture
async def harness(mock_config, tmp_path):
    built = await build_harness(
        mock_config, tmp_path, metadata_patch_prefixes=ALL_LINK_HEAL_PREFIXES,
    )
    yield built
    await built.journal.close()


def corpus_basis(
    verdict: str,
    *,
    child: str = CHILD,
    parent: str = PARENT,
    child_text: str = CHILD_TEXT,
    parent_text: str = PARENT_TEXT,
    key: str = 'H001',
    project: str = DF,
) -> LinkBasis:
    return LinkBasis(
        project_id=project,
        child_id=child,
        parent_id=parent,
        verdict=Verdict(verdict),
        child_sha256=text_sha256(child_text),
        parent_sha256=text_sha256(parent_text),
        source=BasisSource.CORPUS,
        key=key,
    )


def seed_link(
    harness: LinkHealHarness,
    *,
    kind: str | None = AMENDMENT_KIND,
    child: str = CHILD,
    parent: str = PARENT,
    child_text: str = CHILD_TEXT,
    parent_text: str | None = PARENT_TEXT,
    project: str = DF,
    contested: bool = False,
) -> None:
    if parent_text is not None:
        harness.seed(project, parent, parent_text)
    meta: dict[str, object] = {PARENT_ID_KEY: parent}
    if kind is not None:
        meta['kind'] = kind
    if contested:
        meta[CONTESTED_METADATA_KEY] = True
    harness.seed(project, child, child_text, **meta)


async def plan(harness: LinkHealHarness, *bases: LinkBasis) -> Plan:
    return await build_plan(
        bases, store=harness.store(), census=harness.mem0, projects=PROJECTS,
    )


class TestPlannedHeals:
    @pytest.mark.asyncio
    async def test_a_corpus_rated_misfiled_amendment_plans_one_detach(self, harness):
        seed_link(harness)
        basis = corpus_basis('RELATED')

        result = await plan(harness, basis)

        (action,) = result.actions
        assert action.project_id == DF
        assert action.child_id == CHILD
        assert action.action is HealAction.DETACH
        assert action.pre_image == LinkImage(parent_id=PARENT, kind=AMENDMENT_KIND)
        assert action.post_image == LinkImage(parent_id=None)
        assert action.basis_source is BasisSource.CORPUS
        assert action.basis_key == 'H001'
        assert action.child_sha256 == basis.child_sha256
        assert action.parent_sha256 == basis.parent_sha256
        counts = result.counts
        assert (counts.links_total, counts.examined, counts.adjudicated) == (1, 1, 1)
        assert counts.planned == 1
        assert dict(counts.planned_by_action) == {HealAction.DETACH.value: 1}

    @pytest.mark.asyncio
    async def test_an_unrated_link_to_a_deleted_parent_is_a_deterministic_detach(self, harness):
        seed_link(harness, parent_text=None)

        result = await plan(harness)

        (action,) = result.actions
        assert action.action is HealAction.DETACH
        assert action.basis_source is BasisSource.DETERMINISTIC
        assert action.child_sha256 == text_sha256(CHILD_TEXT)
        assert action.parent_sha256 is None
        assert result.counts.planned == 1
        assert result.counts.adjudicated == 0
        assert result.counts.unexamined == 0


class TestReportedLinks:
    @pytest.mark.asyncio
    async def test_a_parent_present_only_in_another_run_project_is_reported(self, harness):
        seed_link(harness, parent_text=None)
        harness.seed(REIFY, PARENT, PARENT_TEXT)

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.cross_project_reported == 1
        assert result.counts.planned == 0

    @pytest.mark.asyncio
    async def test_a_parent_that_is_itself_a_link_is_a_chain(self, harness):
        harness.seed(DF, GRANDPARENT, 'the grandparent note')
        seed_link(harness, parent_text=None)
        harness.seed(
            DF, PARENT, PARENT_TEXT, **{PARENT_ID_KEY: GRANDPARENT, 'kind': AMENDMENT_KIND},
        )

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.chain_reported == 1
        assert result.counts.unexamined == 1  # the parent's own, unrated link

    @pytest.mark.asyncio
    async def test_a_contested_child_rated_related_is_reported_not_detached(self, harness):
        seed_link(harness, contested=True)

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.contested_reported == 1

    @pytest.mark.asyncio
    async def test_a_child_edited_since_export_is_a_stale_rating(self, harness):
        seed_link(harness, child_text='the child note, edited after export')

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.stale_rating == 1
        assert result.counts.adjudicated == 0

    @pytest.mark.asyncio
    async def test_a_corpus_row_whose_child_moved_parent_is_unlinked(self, harness):
        harness.seed(DF, PARENT, PARENT_TEXT)
        seed_link(harness, parent=OTHER_PARENT, parent_text='another parent note')

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.unexamined == 1
        assert result.counts.bases_unlinked == 1
        assert result.counts.adjudicated == 0

    @pytest.mark.asyncio
    async def test_an_unrated_live_link_is_unexamined(self, harness):
        seed_link(harness)

        result = await plan(harness)

        assert result.actions == ()
        assert result.counts.unexamined == 1
        assert result.counts.bases_unlinked == 0

    @pytest.mark.asyncio
    async def test_a_half_link_with_a_child_rated_extends_has_children(self, harness):
        seed_link(harness, kind=None)
        harness.seed(
            DF, GRANDCHILD, 'a note filed under the child',
            **{PARENT_ID_KEY: CHILD, 'kind': SIGHTING_KIND},
        )

        result = await plan(harness, corpus_basis('EXTENDS'))

        assert result.actions == ()
        assert result.counts.has_children == 1
        assert result.counts.chain_reported == 1  # the grandchild's link onto a link


class TestReadFailures:
    @pytest.mark.asyncio
    async def test_a_parent_read_timeout_is_read_failed_and_plans_nothing(self, harness):
        seed_link(harness)
        harness.mem0.read_failures.add(PARENT)

        result = await plan(harness, corpus_basis('RELATED'))

        assert result.actions == ()
        assert result.counts.read_failed == 1
        assert result.counts.examined == 0
        assert result.counts.links_total == 1
        assert result.counts.bases_unlinked == 0


REPORT_COUNTERS = (
    'stale_rating', 'contested_reported', 'chain_reported', 'peer_reported',
    'has_children', 'cross_project_reported', 'unclear', 'no_action', 'unexamined',
)


class TestDisclosureIdentities:
    async def _mixed_store(self, harness: LinkHealHarness) -> Plan:
        seed_link(harness)
        seed_link(
            harness, child=SECOND_CHILD, parent=OTHER_PARENT, kind=SIGHTING_KIND,
            child_text='a second child', parent_text='a second parent',
        )
        harness.seed(
            DF, GRANDCHILD, 'an orphaned note',
            **{PARENT_ID_KEY: GRANDPARENT, 'kind': SIGHTING_KIND},
        )
        harness.seed(DF, '77777777-7777-4777-8777-777777777777', 'an unread note',
                     **{PARENT_ID_KEY: '88888888-8888-4888-8888-888888888888'})
        harness.mem0.read_failures.add('77777777-7777-4777-8777-777777777777')
        return await plan(
            harness,
            corpus_basis('RELATED'),
            corpus_basis(
                'EXTENDS', child=SECOND_CHILD, parent=OTHER_PARENT, key='H002',
                child_text='a second child', parent_text='a second parent',
            ),
            corpus_basis('SAME', child=GRANDPARENT, key='H003'),
        )

    @pytest.mark.asyncio
    async def test_every_link_is_examined_or_read_failed(self, harness):
        counts = (await self._mixed_store(harness)).counts

        assert counts.links_total == 4
        assert counts.links_total == counts.examined + counts.read_failed
        assert counts.read_failed == 1

    @pytest.mark.asyncio
    async def test_every_examined_link_has_exactly_one_disposition(self, harness):
        counts = (await self._mixed_store(harness)).counts

        reported = sum(getattr(counts, name) for name in REPORT_COUNTERS)
        assert counts.examined == counts.planned + reported

    @pytest.mark.asyncio
    async def test_adjudicated_counts_the_links_with_a_current_basis(self, harness):
        counts = (await self._mixed_store(harness)).counts

        assert counts.adjudicated == 2
        assert counts.bases_unlinked == 1

    @pytest.mark.asyncio
    async def test_planned_matches_the_actions_by_action(self, harness):
        result = await self._mixed_store(harness)

        assert result.counts.planned == len(result.actions) == 3
        by_action: dict[str, int] = {}
        for action in result.actions:
            by_action[action.action.value] = by_action.get(action.action.value, 0) + 1
        assert dict(result.counts.planned_by_action) == by_action == {
            HealAction.DETACH.value: 2, HealAction.RELABEL.value: 1,
        }

    @pytest.mark.asyncio
    async def test_planning_writes_nothing(self, harness):
        await self._mixed_store(harness)

        assert harness.mem0.write_count == 0
        assert harness.journal_rows() == []


class TestRunCounts:
    def test_a_fresh_disclosure_is_complete(self):
        assert RunCounts().complete is True

    @pytest.mark.parametrize(
        'partial',
        [
            {'read_failed': 1},
            {'failed': 1},
            {'skipped_cap': 1},
            {'not_attempted': 1},
            {'stopped_by': 'write_failure_streak'},
        ],
        ids=lambda partial: next(iter(partial)),
    )
    def test_a_partial_run_is_never_complete(self, partial):
        assert RunCounts(**partial).complete is False

    def test_skipped_stale_alone_is_still_complete(self):
        assert RunCounts(skipped_stale=3, applied=1).complete is True

    def test_as_json_is_plain_json_carrying_the_completeness(self):
        counts = RunCounts(
            planned=2,
            planned_by_action={HealAction.DETACH.value: 2},
            would_escape=('link-heal-backlog',),
            escaped=({'anchor': 'link-heal-backlog', 'escalation_id': 'esc-1'},),
            caps_bit=('max_actions_per_run',),
            skipped_cap=1,
        )

        document = json.loads(json.dumps(counts.as_json()))

        assert document['planned_by_action'] == {'detach': 2}
        assert document['would_escape'] == ['link-heal-backlog']
        assert document['escaped'] == [
            {'anchor': 'link-heal-backlog', 'escalation_id': 'esc-1'},
        ]
        assert document['caps_bit'] == ['max_actions_per_run']
        assert document['complete'] is False
        assert document['stopped_by'] is None
