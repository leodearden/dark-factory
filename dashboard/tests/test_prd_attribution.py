"""Tests for ``dashboard.data.prd_attribution``: the pure parse of a task's
metadata into PRD provenance and lineage links (PRD dashboard-prd-lanes §4.1–§4.3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dashboard.data.prd_attribution import (
    LINEAGE_METADATA_KEYS,
    MAX_HOPS,
    Attribution,
    LineageEntry,
    LinkField,
    LocalLink,
    PrdKey,
    UnresolvableLink,
    Unresolved,
    UnresolvedReason,
    attribute,
    coalesce_prd,
    lineage_entry,
    normalise_prd_key,
    parse_link,
)

ROOT = '/home/leo/src/dark-factory'

# ---------------------------------------------------------------------------
# coalesce_prd: one string of PRD provenance out of three legacy keys
# ---------------------------------------------------------------------------


def test_coalesce_prd_from_prd_path():
    assert coalesce_prd({'prd_path': 'plans/dashboard-taskgraph-legibility-prd.md'}) == (
        'plans/dashboard-taskgraph-legibility-prd.md'
    )


def test_coalesce_prd_strips_anchor_suffix():
    assert coalesce_prd({'prd_path': 'plans/foo-prd.md#implementation-notes'}) == (
        'plans/foo-prd.md'
    )


def test_coalesce_prd_strips_section_suffix():
    assert coalesce_prd({'prd_path': 'plans/foo-prd.md§Contract'}) == 'plans/foo-prd.md'


def test_coalesce_prd_trims_whitespace():
    assert coalesce_prd({'prd_path': '  plans/foo-prd.md  '}) == 'plans/foo-prd.md'


def test_coalesce_prd_legacy_prd_key():
    assert coalesce_prd({'prd': 'docs/legacy-prd.md'}) == 'docs/legacy-prd.md'


def test_coalesce_prd_legacy_prd_ref_key():
    assert coalesce_prd({'prd_ref': 'docs/legacy-ref-prd.md'}) == 'docs/legacy-ref-prd.md'


def test_coalesce_prd_precedence_prd_path_over_prd_and_ref():
    metadata = {
        'prd_path': 'plans/winner-prd.md',
        'prd': 'plans/loser-prd.md',
        'prd_ref': 'plans/loser-ref-prd.md',
    }
    assert coalesce_prd(metadata) == 'plans/winner-prd.md'


def test_coalesce_prd_empty_prd_path_falls_through_to_prd():
    assert coalesce_prd({'prd_path': '', 'prd': 'plans/fallback-prd.md'}) == (
        'plans/fallback-prd.md'
    )


def test_coalesce_prd_suffix_only_prd_path_falls_through():
    metadata = {'prd_path': '#just-an-anchor', 'prd': 'plans/fallback-prd.md'}
    assert coalesce_prd(metadata) == 'plans/fallback-prd.md'


def test_coalesce_prd_none_when_no_provenance_keys():
    assert coalesce_prd({}) is None


def test_coalesce_prd_non_string_values_skipped():
    assert coalesce_prd({'prd_path': 123}) is None
    assert coalesce_prd({'prd_path': None}) is None


# ---------------------------------------------------------------------------
# normalise_prd_key: one key per PRD file, whichever way a task spelled it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'raw',
    ['plans/x-prd.md', './plans/x-prd.md', f'{ROOT}/plans/x-prd.md'],
    ids=['relative', 'dot_relative', 'absolute_under_root'],
)
def test_normalise_prd_key_resolves_one_key(raw):
    assert normalise_prd_key(raw, ROOT) == PrdKey('plans/x-prd.md')


@pytest.mark.parametrize(
    'raw',
    ['/elsewhere/plans/x-prd.md', '../reify/plans/y-prd.md'],
    ids=['absolute_elsewhere', 'relative_escape'],
)
def test_normalise_prd_key_foreign(raw):
    assert normalise_prd_key(raw, ROOT) == Unresolved(UnresolvedReason.FOREIGN_PRD)


# ---------------------------------------------------------------------------
# parse_link: THE one parser of a task's lineage link
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('value', [12, ' 12 '], ids=['int', 'padded_digits'])
def test_parse_link_local(value):
    assert parse_link({'spawned_from': value}) == LocalLink(LinkField.SPAWNED_FROM, 12)


def test_parse_link_cross_project():
    assert parse_link({'spawned_from': 'reify:5569'}) == UnresolvableLink(
        LinkField.SPAWNED_FROM, UnresolvedReason.CROSS_PROJECT_LINK, 'reify:5569',
    )


@pytest.mark.parametrize(
    'value',
    [{'id': 12}, 'cand-12', True, 0, -3],
    ids=['dict', 'candidate_string', 'bool', 'zero', 'negative'],
)
def test_parse_link_unparseable(value):
    link = parse_link({'x_recovered_from_task': value})
    assert isinstance(link, UnresolvableLink)
    assert link.field is LinkField.X_RECOVERED_FROM_TASK
    assert link.reason is UnresolvedReason.UNPARSEABLE_LINK


def test_parse_link_none_without_link_key():
    assert parse_link({'prd_path': 'plans/x-prd.md'}) is None


def test_parse_link_takes_the_first_present_field_in_precedence_order():
    link = parse_link({'cross_repo_refile_of': 40, 'spawned_from': 7})
    assert link == LocalLink(LinkField.SPAWNED_FROM, 7)


def test_parse_link_skips_a_none_valued_field():
    link = parse_link({'spawned_from': None, 'cross_repo_refile_of': 40})
    assert link == LocalLink(LinkField.CROSS_REPO_REFILE_OF, 40)


def test_unresolvable_link_refuses_a_reason_it_cannot_carry():
    with pytest.raises(ValueError, match='missing_target'):
        UnresolvableLink(LinkField.SPAWNED_FROM, UnresolvedReason.MISSING_TARGET, '12')


# ---------------------------------------------------------------------------
# lineage_entry: the single parse of one task's metadata
# ---------------------------------------------------------------------------


def test_lineage_entry_without_prd_or_link():
    assert lineage_entry({}, ROOT) == LineageEntry(
        own_prd=None, link=None, multi_link=False, done_stamped_at=None, done_kind=None,
    )


def test_lineage_entry_own_prd_is_normalised():
    entry = lineage_entry({'prd_path': f'{ROOT}/plans/x-prd.md#notes'}, ROOT)
    assert entry.own_prd == PrdKey('plans/x-prd.md')


def test_lineage_entry_foreign_own_prd():
    entry = lineage_entry({'prd': '/elsewhere/plans/x-prd.md'}, ROOT)
    assert entry.own_prd == Unresolved(UnresolvedReason.FOREIGN_PRD)


def test_lineage_entry_link_is_parsed():
    entry = lineage_entry({'spawned_from': '12'}, ROOT)
    assert entry.link == LocalLink(LinkField.SPAWNED_FROM, 12)
    assert entry.multi_link is False


def test_lineage_entry_multi_link_when_two_link_keys_present():
    entry = lineage_entry({'spawned_from': 12, 'cross_repo_refile_of': 'reify:5'}, ROOT)
    assert entry.multi_link is True
    assert entry.link == LocalLink(LinkField.SPAWNED_FROM, 12)


def test_lineage_entry_done_stamped_at_parsed():
    entry = lineage_entry(
        {'done_provenance': {'kind': 'merged', 'stamped_at': '2026-11-01T10:00:00+00:00'}},
        ROOT,
    )
    assert entry.done_stamped_at == datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    'done_provenance',
    ['merged', {'kind': 'merged'}, {'stamped_at': 17}, {'stamped_at': 'not-a-time'}],
    ids=['non_dict', 'no_stamp', 'non_str_stamp', 'unparseable_stamp'],
)
def test_lineage_entry_done_stamped_at_none(done_provenance):
    entry = lineage_entry({'done_provenance': done_provenance}, ROOT)
    assert entry.done_stamped_at is None


def test_lineage_entry_done_kind_is_the_provenance_kind():
    entry = lineage_entry(
        {'done_provenance': {'kind': 'merged', 'stamped_at': '2026-11-01T10:00:00+00:00'}},
        ROOT,
    )
    assert entry.done_kind == 'merged'
    assert entry.done_stamped_at == datetime(2026, 11, 1, 10, 0, tzinfo=UTC)


def test_lineage_entry_done_kind_without_a_stamp():
    entry = lineage_entry({'done_provenance': {'kind': 'operator'}}, ROOT)
    assert entry.done_kind == 'operator'
    assert entry.done_stamped_at is None


@pytest.mark.parametrize(
    'metadata',
    [
        {},
        {'done_provenance': 'merged'},
        {'done_provenance': {'stamped_at': '2026-11-01T10:00:00+00:00'}},
        {'done_provenance': {'kind': 7}},
        {'done_provenance': {'kind': ''}},
    ],
    ids=['absent', 'non_dict', 'no_kind', 'non_str_kind', 'empty_kind'],
)
def test_lineage_entry_done_kind_none(metadata):
    assert lineage_entry(metadata, ROOT).done_kind is None


def test_lineage_metadata_keys_pin_the_read():
    assert LINEAGE_METADATA_KEYS == (
        'prd_path', 'prd', 'prd_ref',
        'spawned_from', 'x_recovered_from_task', 'cross_repo_refile_of',
        'done_provenance',
    )


# ---------------------------------------------------------------------------
# attribute: every task's PRD, by its own provenance or its lineage chain
# ---------------------------------------------------------------------------


def _entries(metadata_by_id: dict[int, dict]) -> dict[int, LineageEntry]:
    return {task_id: lineage_entry(metadata, ROOT) for task_id, metadata in metadata_by_id.items()}


def _chain(head: int, length: int, prd: str) -> dict[int, dict]:
    """``length`` links from *head* down to a root carrying *prd*: ids head..head-length."""
    root = head - length
    chain: dict[int, dict] = {root: {'prd_path': prd}}
    for task_id in range(root + 1, head + 1):
        chain[task_id] = {'spawned_from': task_id - 1}
    return chain


_FIXTURE: dict[int, dict] = {
    1: {'prd_path': 'plans/a-prd.md'},
    **{
        task_id: metadata
        for length in range(1, 7)
        for task_id, metadata in _chain(100 * length + length, length, f'plans/chain-{length}-prd.md').items()
    },
    20: {'spawned_from': 21},
    21: {'spawned_from': 20},
    22: {'spawned_from': 20},
    **_chain(909, 9, 'plans/deep-prd.md'),
    30: {'spawned_from': 'reify:5569'},
    31: {'spawned_from': {'id': 3}},
    32: {'x_recovered_from_task': 'cand-12'},
    33: {'spawned_from': 99999},
    34: {},
    35: {'spawned_from': 36},
    36: {},
    37: {'prd': '/elsewhere/plans/x-prd.md', 'spawned_from': 1},
    38: {'spawned_from': 37},
    39: {'spawned_from': 1, 'cross_repo_refile_of': 'reify:4'},
}


def _attributed() -> tuple[dict[int, Attribution], int]:
    attributions, multi_link = attribute(_entries(_FIXTURE))
    return dict(attributions), multi_link


def test_max_hops_is_eight():
    assert MAX_HOPS == 8


def test_direct_task_is_attributed_at_hop_zero():
    result = _attributed()[0][1]
    assert result == Attribution(prd='plans/a-prd.md', hops=0, unresolved=None)
    assert result.direct is True


@pytest.mark.parametrize('length', range(1, 7))
def test_chain_resolves_to_its_root_prd(length):
    head = 100 * length + length
    result = _attributed()[0][head]
    assert result == Attribution(prd=f'plans/chain-{length}-prd.md', hops=length, unresolved=None)
    assert result.direct is False


def test_two_cycle_and_a_task_leading_into_it():
    attributions = _attributed()[0]
    assert attributions[20] == Attribution(prd=None, hops=1, unresolved=UnresolvedReason.CYCLE)
    assert attributions[21] == Attribution(prd=None, hops=1, unresolved=UnresolvedReason.CYCLE)
    assert attributions[22] == Attribution(prd=None, hops=2, unresolved=UnresolvedReason.CYCLE)


def test_nine_hop_chain_exceeds_depth_while_eight_hops_resolve():
    attributions = _attributed()[0]
    assert attributions[909] == Attribution(
        prd=None, hops=MAX_HOPS, unresolved=UnresolvedReason.DEPTH_EXCEEDED,
    )
    assert attributions[908] == Attribution(prd='plans/deep-prd.md', hops=8, unresolved=None)


@pytest.mark.parametrize(
    ('task_id', 'expected'),
    [
        (30, Attribution(prd=None, hops=0, unresolved=UnresolvedReason.CROSS_PROJECT_LINK)),
        (31, Attribution(prd=None, hops=0, unresolved=UnresolvedReason.UNPARSEABLE_LINK)),
        (32, Attribution(prd=None, hops=0, unresolved=UnresolvedReason.UNPARSEABLE_LINK)),
        (33, Attribution(prd=None, hops=0, unresolved=UnresolvedReason.MISSING_TARGET)),
        (34, Attribution(prd=None, hops=0, unresolved=UnresolvedReason.NO_LINK)),
        (35, Attribution(prd=None, hops=1, unresolved=UnresolvedReason.ROOT_WITHOUT_PRD)),
        (36, Attribution(prd=None, hops=0, unresolved=UnresolvedReason.NO_LINK)),
        (37, Attribution(prd=None, hops=0, unresolved=UnresolvedReason.FOREIGN_PRD)),
        (38, Attribution(prd=None, hops=1, unresolved=UnresolvedReason.FOREIGN_PRD)),
        (39, Attribution(prd='plans/a-prd.md', hops=1, unresolved=None)),
    ],
    ids=[
        'cross_project', 'dict_link', 'candidate_link', 'missing_target', 'no_link',
        'root_without_prd', 'bare_root', 'own_foreign_prd', 'foreign_ancestor',
        'multi_link_follows_precedence',
    ],
)
def test_unresolved_and_edge_cases(task_id, expected):
    assert _attributed()[0][task_id] == expected


def test_a_task_carrying_two_link_keys_counts_once():
    assert _attributed()[1] == 1


def test_every_attribution_is_direct_exactly_at_hop_zero():
    for result in _attributed()[0].values():
        assert result.direct is (result.hops == 0)


def test_every_task_in_the_fixture_is_attributed():
    assert set(_attributed()[0]) == set(_FIXTURE)


@pytest.mark.parametrize('descending', [False, True], ids=['ascending', 'descending'])
def test_ten_node_chain_is_attributed_independently_of_iteration_order(descending):
    ids = sorted(range(1, 11), reverse=descending)
    metadata: dict[int, dict] = {task_id: {'spawned_from': task_id + 1} for task_id in ids}
    metadata[10] = {'prd_path': 'plans/ten-prd.md'}
    ordered = {task_id: metadata[task_id] for task_id in ids}
    attributions = attribute(_entries(ordered))[0]
    assert attributions[2] == Attribution(prd='plans/ten-prd.md', hops=8, unresolved=None)
    assert attributions[1] == Attribution(
        prd=None, hops=MAX_HOPS, unresolved=UnresolvedReason.DEPTH_EXCEEDED,
    )
    assert [attributions[task_id].hops for task_id in range(3, 11)] == list(range(7, -1, -1))


def test_whole_fixture_is_attributed_independently_of_iteration_order():
    ascending = dict(attribute(_entries(dict(sorted(_FIXTURE.items()))))[0])
    descending = dict(attribute(_entries(dict(sorted(_FIXTURE.items(), reverse=True))))[0])
    assert ascending == descending


def test_a_missing_target_at_the_depth_limit_exceeds_depth():
    """The walk checks depth before it looks the target up, from every start."""
    for descending in (False, True):
        ids = sorted(range(1, 10), reverse=descending)
        metadata = {task_id: {'spawned_from': task_id + 1} for task_id in ids}
        metadata[9] = {'spawned_from': 99999}
        attributions = attribute(_entries({task_id: metadata[task_id] for task_id in ids}))[0]
        assert attributions[1] == Attribution(
            prd=None, hops=MAX_HOPS, unresolved=UnresolvedReason.DEPTH_EXCEEDED,
        )
        assert attributions[2] == Attribution(
            prd=None, hops=7, unresolved=UnresolvedReason.MISSING_TARGET,
        )


@pytest.mark.parametrize(
    ('kwargs', 'named'),
    [
        ({'prd': 'k', 'hops': 0, 'unresolved': UnresolvedReason.CYCLE}, 'cycle'),
        ({'prd': None, 'hops': 0, 'unresolved': None}, 'prd'),
        ({'prd': 'k', 'hops': 9, 'unresolved': None}, '9'),
    ],
    ids=['prd_and_reason', 'neither', 'hops_over_max'],
)
def test_attribution_refuses_an_impossible_value(kwargs, named):
    with pytest.raises(ValueError, match=named):
        Attribution(**kwargs)
