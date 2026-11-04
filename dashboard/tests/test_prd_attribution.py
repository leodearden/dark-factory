"""Tests for ``dashboard.data.prd_attribution``: the pure parse of a task's
metadata into PRD provenance and lineage links (PRD dashboard-prd-lanes §4.1–§4.3)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dashboard.data.prd_attribution import (
    LINEAGE_METADATA_KEYS,
    LineageEntry,
    LinkField,
    LocalLink,
    PrdKey,
    UnresolvableLink,
    Unresolved,
    UnresolvedReason,
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
        own_prd=None, link=None, multi_link=False, done_stamped_at=None,
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


def test_lineage_metadata_keys_pin_the_read():
    assert LINEAGE_METADATA_KEYS == (
        'prd_path', 'prd', 'prd_ref',
        'spawned_from', 'x_recovered_from_task', 'cross_repo_refile_of',
        'done_provenance',
    )
