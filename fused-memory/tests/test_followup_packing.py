"""The held-follow-up pack generator: plans/followup-packing-sweep-prd.md §4.1.

Synthetic members pin each rule. The 2026-11-04 pool dump and the a2+c2 CSV the re-run produced
from it (copied verbatim under ``fixtures/``) pin the whole generator: built from every dump
record, ``candidate_packs`` must reproduce that CSV.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from shared.followup_items import MAX_FOLLOWUP_ITEMS
from shared.task_statuses import TaskStatus

from fused_memory.middleware.followup_packing import (
    HUB_SHARE,
    PACK_MAX_ITEMS,
    PACK_MAX_UNION_KEYS,
    RIDER_RESERVE_DAYS,
    RIDER_RESERVE_MAX_DECLINES,
    CandidatePack,
    PackExclusion,
    PackMember,
    PackRecipe,
    candidate_packs,
    exclusion,
    hub_keys,
    member_text_fp,
    pack_lock_keys,
    rider_reserved,
    survivor_of,
    title_recipe,
)

NOW = datetime(2026, 11, 4, tzinfo=UTC)
FIXTURES = Path(__file__).parent / 'fixtures'
POOL_DUMP = FIXTURES / 'pack_pool_2026-11-04.json'
PACKS_CSV = FIXTURES / 'packs-a2c2-2026-11-04.csv'
PROJECTS = ('dark_factory', 'reify')


def member(**overrides: Any) -> PackMember:
    """A member no exclusion applies to, with *overrides* applied."""
    title = overrides.get('title', f'Tidy task {overrides.get("task_id", 10)}')
    fields: dict[str, Any] = {
        'task_id': 10,
        'title': title,
        'kind': 'followup',
        'parent': 1,
        'lock_keys': (),
        'item_count': 1,
        'item_keys': (f'k{overrides.get("task_id", 10)}',),
        'band': None,
        'pending_since': NOW - timedelta(days=30),
        'text_fp': member_text_fp(title, '', ''),
        'status': TaskStatus.PENDING,
        'dependencies': frozenset(),
        'held': True,
        'is_holder': False,
        'has_live_dependents': False,
        'claimed': False,
        'has_branch': False,
        'has_plan': False,
        'dedupe_skipped': False,
        'rider_of': None,
        'rider_declines': 0,
        'rider_last_decline_reason': None,
    }
    return PackMember(**{**fields, **overrides})


@dataclass(frozen=True)
class ProjectPool:
    header: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    members: tuple[PackMember, ...]


def _parsed_pending_since(raw: str) -> datetime | None:
    return datetime.fromisoformat(raw.replace('Z', '+00:00')) if raw else None


def _dump_member(record: Mapping[str, Any], *, lock_depth: int) -> PackMember:
    """A dump record as a member, every field the dump lacks forced to its no-exclusion value."""
    parent = record['parent']
    return PackMember(
        task_id=record['task_id'],
        title=record['title'],
        kind=record['kind'],
        parent=parent if isinstance(parent, int) else None,
        lock_keys=pack_lock_keys(record['files'], lock_depth=lock_depth),
        item_count=record['item_count'],
        item_keys=(f'k{record["task_id"]}',),
        band=None,
        pending_since=_parsed_pending_since(record['pending_since']),
        text_fp=member_text_fp(record['title'], '', ''),
        status=TaskStatus.PENDING,
        dependencies=frozenset(),
        held=True,
        is_holder=False,
        has_live_dependents=False,
        claimed=False,
        has_branch=False,
        has_plan=False,
        dedupe_skipped=False,
        rider_of=None,
        rider_declines=0,
        rider_last_decline_reason=None,
    )


@pytest.fixture(scope='module')
def pool() -> dict[str, ProjectPool]:
    dump = json.loads(POOL_DUMP.read_text())
    pools = {}
    for project in PROJECTS:
        header = dump['header']['projects'][project]
        records = tuple(r for r in dump['members'] if r['project'] == project)
        members = tuple(_dump_member(r, lock_depth=header['lock_depth']) for r in records)
        pools[project] = ProjectPool(header=header, records=records, members=members)
    return pools


def test_bounds_have_one_home_each():
    assert PACK_MAX_ITEMS == MAX_FOLLOWUP_ITEMS == 5
    assert PACK_MAX_UNION_KEYS == 15
    assert HUB_SHARE == 0.02
    assert RIDER_RESERVE_DAYS == 14
    assert RIDER_RESERVE_MAX_DECLINES == 3


@pytest.mark.parametrize(
    ('overrides', 'field', 'value'),
    [
        ({'pending_since': datetime(2026, 10, 1)}, 'pending_since', '2026-10-01'),
        ({'kind': 'chore'}, 'kind', 'chore'),
        ({'band': 'huge'}, 'band', 'huge'),
        ({'item_count': 0}, 'item_count', '0'),
        ({'lock_keys': ('b/y.py', 'a/x.py')}, 'lock_keys', 'b/y.py'),
        ({'lock_keys': ('a/x.py', 'a/x.py')}, 'lock_keys', 'a/x.py'),
        ({'status': 'pending-ish'}, 'status', 'pending-ish'),
    ],
)
def test_member_refuses_a_value_outside_its_vocabulary(overrides, field, value):
    with pytest.raises(ValueError, match=field) as refused:
        member(**overrides)
    assert value in str(refused.value)


@pytest.mark.parametrize('field', ['title', 'status'])
def test_member_is_frozen(field: str):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(member(), field, 'renamed')


def test_pack_lock_keys_drops_directory_charters_and_normalises_to_the_lock_depth():
    files = ['b/y/deep/z.py', 'orchestrator/src', 'a/x.py', 'b/y/deep/w.py', 'a/x.py']
    assert pack_lock_keys(files, lock_depth=3) == ('a/x.py', 'b/y/deep')


@pytest.mark.parametrize('project', PROJECTS)
def test_pack_lock_keys_reproduces_the_dump_lock_keys(pool, project):
    depth = pool[project].header['lock_depth']
    mismatched = [
        r['task_id']
        for r in pool[project].records
        if pack_lock_keys(r['files'], lock_depth=depth) != tuple(r['lock_keys'])
    ]
    assert mismatched == []


@pytest.mark.parametrize(('project', 'held'), [('dark_factory', 1405), ('reify', 1112)])
def test_dump_members_are_the_whole_held_pool(pool, project, held):
    assert len(pool[project].members) == pool[project].header['held'] == held


def test_member_text_fp_is_a_stable_16_char_hex_digest():
    fp = member_text_fp('title', 'description', 'details')
    assert len(fp) == 16
    assert int(fp, 16) >= 0
    assert member_text_fp('title', 'description', 'details') == fp


@pytest.mark.parametrize(
    'changed', [('Title', 'description', 'details'), ('title', 'Description', 'details'),
                ('title', 'description', 'Details')],
)
def test_member_text_fp_changes_with_any_field(changed):
    assert member_text_fp(*changed) != member_text_fp('title', 'description', 'details')


def test_member_text_fp_keeps_the_field_boundary():
    assert member_text_fp('ab', 'c', '') != member_text_fp('a', 'bc', '')


@pytest.mark.parametrize(
    ('overrides', 'reserved'),
    [
        ({'lock_keys': ('a/x.py',), 'pending_since': NOW - timedelta(days=10)}, True),
        ({'lock_keys': ('a/x.py',), 'pending_since': NOW - timedelta(days=15)}, False),
        ({'lock_keys': ('a/x.py',), 'pending_since': NOW - timedelta(days=14)}, False),
        ({'lock_keys': (), 'pending_since': NOW - timedelta(days=1)}, False),
        ({'lock_keys': ('a/x.py',), 'pending_since': NOW - timedelta(days=10), 'rider_declines': 3}, False),
        ({'lock_keys': ('a/x.py',), 'pending_since': NOW - timedelta(days=10), 'rider_declines': 2}, True),
        ({'lock_keys': ('a/x.py',), 'pending_since': None}, False),
    ],
)
def test_rider_reserved_holds_young_filed_members_back_for_riders(overrides, reserved):
    assert rider_reserved(member(**overrides), now=NOW) is reserved


def test_rider_reserved_refuses_a_naive_now():
    with pytest.raises(ValueError, match='now'):
        rider_reserved(member(), now=datetime(2026, 11, 4))


@pytest.mark.parametrize(('project', 'reserved'), [('dark_factory', 467), ('reify', 169)])
def test_rider_reserved_count_on_the_dump_matches_the_prd_table(pool, project, reserved):
    assert sum(rider_reserved(m, now=NOW) for m in pool[project].members) == reserved


@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [
        ({'is_holder': True}, PackExclusion.HOLDER),
        ({'held': False}, PackExclusion.NOT_HELD),
        ({'status': TaskStatus.IN_PROGRESS}, PackExclusion.NOT_HELD),
        ({'band': 'broad'}, PackExclusion.BROAD),
        ({'has_live_dependents': True}, PackExclusion.LIVE_DEPENDENTS),
        ({'claimed': True}, PackExclusion.CLAIMED),
        ({'has_branch': True}, PackExclusion.HAS_BRANCH),
        ({'has_plan': True}, PackExclusion.HAS_PLAN),
        ({'dedupe_skipped': True}, PackExclusion.DEDUPE_SKIPPED),
        ({'rider_of': 9}, PackExclusion.RIDER),
        ({'lock_keys': ('a/x.py',), 'pending_since': NOW - timedelta(days=10)}, PackExclusion.RIDER_RESERVED),
    ],
)
def test_exclusion_names_the_one_condition_set(overrides, expected):
    assert exclusion(member(**overrides), now=NOW) is expected


@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [
        ({'is_holder': True, 'band': 'broad'}, PackExclusion.HOLDER),
        ({'claimed': True, 'rider_of': 9}, PackExclusion.CLAIMED),
    ],
)
def test_exclusion_first_match_wins_in_declaration_order(overrides, expected):
    assert exclusion(member(**overrides), now=NOW) is expected


@pytest.mark.parametrize('band', [None, 'undeclared', 'narrow', 'moderate'])
def test_exclusion_admits_every_band_but_broad(band):
    assert exclusion(member(band=band), now=NOW) is None


def test_a_decline_reason_is_a_hint_never_an_exclusion():
    assert exclusion(member(rider_last_decline_reason='different_intent'), now=NOW) is None


HUB = 'h/hub.py'


def _pool_declaring(key: str, *, declaring: int, size: int, fileless: int = 0) -> list[PackMember]:
    with_files = [
        member(task_id=i, lock_keys=(key,) if i < declaring else (f'own/{i}.py',)) for i in range(size)
    ]
    return with_files + [member(task_id=size + i) for i in range(fileless)]


def test_a_key_declared_by_more_than_the_share_is_a_hub():
    assert hub_keys(_pool_declaring(HUB, declaring=2, size=50)) == frozenset({HUB})


def test_a_key_declared_by_exactly_the_share_is_not_a_hub():
    assert hub_keys(_pool_declaring(HUB, declaring=2, size=100)) == frozenset()


def test_fileless_members_count_in_the_hub_denominator():
    assert hub_keys(_pool_declaring(HUB, declaring=2, size=50, fileless=100)) == frozenset()


def test_a_member_counts_once_toward_a_hub_whatever_its_item_count():
    members = _pool_declaring(HUB, declaring=1, size=99) + [
        member(task_id=200, lock_keys=(HUB,), item_count=3),
    ]
    assert hub_keys(members) == frozenset()


@pytest.mark.parametrize('share', [0.0, 1.0, -0.1, 1.5])
def test_hub_keys_refuses_a_share_outside_the_open_unit_interval(share):
    with pytest.raises(ValueError, match='share'):
        hub_keys(_pool_declaring(HUB, declaring=2, size=50), share=share)


def test_an_empty_pool_has_no_hubs():
    assert hub_keys([]) == frozenset()


@pytest.mark.parametrize(('project', 'hubs'), [('dark_factory', 8), ('reify', 16)])
def test_hub_keys_over_the_whole_dump_pool_are_the_header_hubs(pool, project, hubs):
    found = hub_keys(pool[project].members)
    assert found == frozenset(pool[project].header['hub_keys'])
    assert len(found) == hubs


@pytest.mark.parametrize(
    ('title', 'recipe'),
    [
        ('Relocate the inline test mod of foo.rs into its own module', PackRecipe.H14_RELOCATE),
        ('Retarget the stale docstring cites in bar.py', PackRecipe.CITE_DOC_HYGIENE),
        ('Deduplicate the two copies of the parser', PackRecipe.SPOT_DEDUPE),
        ('Stop patching _private by dotted-path', PackRecipe.TEST_SEAM),
        ('Assert the DiagnosticCode instead of a text-match', PackRecipe.DIAG_CODE),
        ('Deduplicate the cite helper', None),
        ('Split engine.rs into its own module and update cites', PackRecipe.H14_RELOCATE),
        ('RELOCATE THE INLINE TEST MOD OF FOO.RS', PackRecipe.H14_RELOCATE),
        ('Add a retry to the fetcher', None),
    ],
)
def test_title_recipe_is_the_first_matching_row(title, recipe):
    assert title_recipe(title) is recipe


def _recipe_label(title: str) -> str | None:
    recipe = title_recipe(title)
    return None if recipe is None else recipe.value


@pytest.mark.parametrize('project', PROJECTS)
def test_title_recipe_reproduces_the_dump_recipe_labels(pool, project):
    mislabelled = [
        (r['task_id'], r['recipe_v1'])
        for r in pool[project].records
        if _recipe_label(r['title']) != r['recipe_v1']
    ]
    assert mislabelled == []


def test_the_oldest_member_survives():
    members = [
        member(task_id=10, pending_since=NOW - timedelta(days=5)),
        member(task_id=11, pending_since=NOW - timedelta(days=20)),
    ]
    assert survivor_of(members) == 11


def test_the_lowest_id_survives_a_tie():
    members = [member(task_id=12), member(task_id=11)]
    assert survivor_of(members) == 11


def test_an_unknown_age_reads_as_oldest():
    members = [
        member(task_id=10, pending_since=NOW - timedelta(days=100)),
        member(task_id=11, pending_since=None),
    ]
    assert survivor_of(members) == 11


def test_survivor_of_refuses_an_empty_pack():
    with pytest.raises(ValueError, match='member'):
        survivor_of([])


SPOT_TITLE = 'Deduplicate the two copies of X'


def _packed(packs: Iterable[CandidatePack]) -> set[tuple[str, frozenset[int]]]:
    return {(p.generator, frozenset(p.members)) for p in packs}


def _packs(members: Iterable[PackMember], **kwargs: Any) -> tuple[CandidatePack, ...]:
    return candidate_packs(list(members), **{'hubs': frozenset(), 'now': NOW, **kwargs})


def _keys(*keys: str) -> tuple[str, ...]:
    return tuple(sorted(keys))


def test_same_parent_members_sharing_a_key_pack_under_the_older():
    (pack,) = _packs([
        member(task_id=10, lock_keys=('a/x.py',)),
        member(task_id=11, lock_keys=('a/x.py',), pending_since=NOW - timedelta(days=40)),
    ])
    assert (pack.generator, pack.key, pack.members) == ('parent', '1', (10, 11))
    assert pack.union_keys == ('a/x.py',)
    assert pack.survivor == 11


def test_a_parent_pack_joins_members_through_a_chain_of_shared_keys():
    packs = _packs([
        member(task_id=10, lock_keys=('a/x.py',)),
        member(task_id=11, lock_keys=_keys('a/x.py', 'b/y.py')),
        member(task_id=12, lock_keys=('b/y.py',)),
    ])
    assert _packed(packs) == {('parent', frozenset({10, 11, 12}))}


def test_kind_splits_a_parent_group():
    packs = _packs([
        member(task_id=10, lock_keys=('a/x.py',)),
        member(task_id=11, lock_keys=('a/x.py',), kind='review_suggestion'),
    ])
    assert packs == ()


def test_a_hub_key_never_joins_members():
    packs = _packs(
        [
            member(task_id=10, lock_keys=(HUB,)),
            member(task_id=11, lock_keys=(HUB,)),
            member(task_id=12, lock_keys=_keys(HUB, 'c/z.py')),
            member(task_id=13, lock_keys=_keys(HUB, 'c/z.py')),
        ],
        hubs=frozenset({HUB}),
    )
    assert _packed(packs) == {('parent', frozenset({12, 13}))}


def test_unequal_dependency_sets_never_pack_together():
    packs = _packs(
        member(task_id=i, lock_keys=('a/x.py',), dependencies=frozenset({99}) if i < 23 else frozenset())
        for i in range(20, 25)
    )
    assert _packed(packs) == {('parent', frozenset({20, 21, 22})), ('parent', frozenset({23, 24}))}


def test_a_pack_holds_at_most_max_items_items():
    packs = _packs([
        member(task_id=10, lock_keys=('a/x.py',), item_count=3),
        member(task_id=11, lock_keys=('a/x.py',), item_count=3),
        member(task_id=12, parent=2, lock_keys=('b/y.py',), item_count=3),
        member(task_id=13, parent=2, lock_keys=('b/y.py',), item_count=2),
    ])
    assert _packed(packs) == {('parent', frozenset({12, 13}))}


def test_a_pack_holds_at_most_max_union_keys_keys():
    packs = _packs(
        member(task_id=i, lock_keys=_keys('s/shared.py', *(f'p{i}/{n}.py' for n in range(6))))
        for i in (10, 11, 12)
    )
    (pack,) = packs
    assert pack.members == (10, 11)
    assert len(pack.union_keys) == 13


@pytest.mark.parametrize(
    ('oversized', 'bounds'),
    [
        pytest.param({'lock_keys': _keys('a/x.py', *(f'o/{n}.py' for n in range(16)))}, {}, id='keys'),
        pytest.param({'lock_keys': ('a/x.py',), 'item_count': 3}, {'max_items': 2}, id='items'),
    ],
)
def test_a_member_over_a_bound_on_its_own_never_packs_and_parts_its_neighbours(oversized, bounds):
    packs = _packs(
        [
            member(task_id=10, lock_keys=('a/x.py',)),
            member(task_id=11, **oversized),
            member(task_id=12, lock_keys=('a/x.py',)),
            member(task_id=13, lock_keys=('a/x.py',)),
        ],
        **bounds,
    )
    assert _packed(packs) == {('parent', frozenset({12, 13}))}


def test_a_large_group_chunks_in_ascending_id():
    packs = _packs(member(task_id=i, lock_keys=('a/x.py',)) for i in range(10, 17))
    assert _packed(packs) == {
        ('parent', frozenset(range(10, 15))),
        ('parent', frozenset({15, 16})),
    }


def _recipe_trio() -> list[PackMember]:
    return [
        member(task_id=40, parent=140, title=SPOT_TITLE, lock_keys=('b/q.py',)),
        member(task_id=41, parent=141, title=SPOT_TITLE, lock_keys=('a/p.py',)),
        member(task_id=42, parent=142, title=SPOT_TITLE),
    ]


def test_members_applying_one_recipe_pack_across_parents():
    (pack,) = _packs(_recipe_trio())
    assert (pack.generator, pack.key, pack.members) == ('recipe', 'spot_dedupe', (40, 41, 42))


def test_a_recipe_group_walks_by_directory_with_fileless_members_last():
    assert _packed(_packs(_recipe_trio(), max_items=2)) == {('recipe', frozenset({40, 41}))}


def test_parent_packs_take_their_members_before_recipe_packs():
    packs = _packs([
        member(task_id=50, parent=5, title=SPOT_TITLE, lock_keys=('a/x.py',)),
        member(task_id=51, parent=5, title=SPOT_TITLE, lock_keys=('a/x.py',)),
        member(task_id=52, parent=152, title=SPOT_TITLE),
        member(task_id=53, parent=153, title=SPOT_TITLE),
    ])
    assert _packed(packs) == {('parent', frozenset({50, 51})), ('recipe', frozenset({52, 53}))}


def test_excluded_members_never_pack():
    excluded = [
        {'is_holder': True}, {'band': 'broad'}, {'has_live_dependents': True}, {'claimed': True},
        {'has_branch': True}, {'has_plan': True}, {'dedupe_skipped': True}, {'rider_of': 9},
    ]
    packs = _packs(
        [member(task_id=60, lock_keys=('a/x.py',)), member(task_id=61, lock_keys=('a/x.py',))]
        + [member(task_id=62 + n, lock_keys=('a/x.py',), **flags) for n, flags in enumerate(excluded)]
    )
    assert _packed(packs) == {('parent', frozenset({60, 61}))}


def test_rider_reserved_members_wait_and_fileless_ones_go_to_recipes():
    packs = _packs([
        member(task_id=70, lock_keys=('a/x.py',), pending_since=NOW - timedelta(days=10)),
        member(task_id=71, lock_keys=('a/x.py',), pending_since=NOW - timedelta(days=15)),
        member(task_id=72, parent=172, kind='review_suggestion', title=SPOT_TITLE,
               pending_since=NOW - timedelta(days=1)),
        member(task_id=73, lock_keys=('a/x.py',), pending_since=NOW - timedelta(days=10),
               rider_declines=RIDER_RESERVE_MAX_DECLINES),
        member(task_id=74, parent=174, kind='review_suggestion', title=SPOT_TITLE,
               pending_since=NOW - timedelta(days=1)),
    ])
    assert _packed(packs) == {('parent', frozenset({71, 73})), ('recipe', frozenset({72, 74}))}


def test_a_decline_reason_is_recorded_on_the_pack_as_a_hint():
    (hinted,) = _packs([
        member(task_id=10, lock_keys=('a/x.py',)),
        member(task_id=11, lock_keys=('a/x.py',), rider_last_decline_reason='different_intent'),
    ])
    (plain,) = _packs([member(task_id=10, lock_keys=('a/x.py',)), member(task_id=11, lock_keys=('a/x.py',))])
    assert hinted.rider_decline_hints == ((11, 'different_intent'),)
    assert plain.rider_decline_hints == ()


def _trio() -> list[PackMember]:
    return [
        member(task_id=10, lock_keys=_keys('a/x.py', 'c/z.py')),
        member(task_id=11, lock_keys=('a/x.py',), pending_since=NOW - timedelta(days=50)),
        member(task_id=12, lock_keys=_keys('a/x.py', 'b/y.py')),
    ]


def test_a_pack_carries_its_survivor_and_its_union_keys():
    trio = _trio()
    (pack,) = _packs(trio)
    assert pack.survivor == survivor_of(trio) == 11
    assert pack.union_keys == ('a/x.py', 'b/y.py', 'c/z.py')


def test_pack_key_follows_member_text_not_input_order():
    trio = _trio()
    (pack,) = _packs(trio)
    (shuffled,) = _packs(reversed(trio))
    (edited,) = _packs([trio[0], trio[1], dataclasses.replace(trio[2], text_fp=member_text_fp('t', 'd', 'e'))])
    assert shuffled.pack_key == pack.pack_key
    assert edited.pack_key != pack.pack_key


def _ordered_pool() -> list[PackMember]:
    return [
        *(member(task_id=i, lock_keys=('a/x.py',)) for i in (10, 11, 12)),
        *(member(task_id=i, parent=2, lock_keys=('b/y.py',), pending_since=NOW - timedelta(days=60))
          for i in (20, 21)),
        member(task_id=30, parent=3, lock_keys=('c/z.py',), pending_since=None),
        member(task_id=31, parent=3, lock_keys=('c/z.py',)),
        *(member(task_id=i, parent=4, lock_keys=('d/w.py',), pending_since=NOW - timedelta(days=60))
          for i in (40, 41)),
    ]


def test_candidate_packs_is_deterministic_under_input_order():
    shuffled = _ordered_pool()
    random.Random(0).shuffle(shuffled)
    assert _packs(shuffled) == _packs(_ordered_pool())


def test_packs_come_largest_first_then_oldest_survivor_then_lowest_id():
    assert [p.members for p in _packs(_ordered_pool())] == [(10, 11, 12), (30, 31), (20, 21), (40, 41)]


@pytest.mark.parametrize(
    'bounds', [{'max_items': 6}, {'max_items': 0}, {'max_union_keys': 16}, {'max_union_keys': 0}],
)
def test_candidate_packs_refuses_to_loosen_a_bound(bounds):
    (name,) = bounds
    with pytest.raises(ValueError, match=name):
        _packs(_trio(), **bounds)


def test_candidate_packs_refuses_a_repeated_task_id():
    with pytest.raises(ValueError, match='task_id'):
        _packs([member(task_id=10), member(task_id=10)])


def test_candidate_packs_refuses_a_naive_now():
    with pytest.raises(ValueError, match='now'):
        _packs(_trio(), now=datetime(2026, 11, 4))


@pytest.mark.parametrize(
    ('members', 'survivor', 'field'),
    [((10,), 10, 'members'), ((11, 10), 10, 'members'), ((10, 11), 12, 'survivor')],
)
def test_candidate_pack_refuses_an_ill_formed_pack(members, survivor, field):
    with pytest.raises(ValueError, match=field):
        CandidatePack('parent', '1', members, ('a/x.py',), survivor=survivor, pack_key='0' * 16)


_CSV_GENERATOR = {'a2': 'parent', 'c2': 'recipe'}
_FIXTURE_RUNS = [
    ('dark_factory', 'whole_pool', 77, 41),
    ('dark_factory', 'minus_reservation', 34, 30),
    ('reify', 'whole_pool', 89, 45),
    ('reify', 'minus_reservation', 76, 36),
]


def _expected_packs(project: str, run: str) -> dict[tuple[str, frozenset[int]], str]:
    with PACKS_CSV.open(newline='') as rows:
        return {
            (_CSV_GENERATOR[r['generator']], frozenset(int(t) for t in r['task_ids'].split())): r['key']
            for r in csv.DictReader(rows)
            if (r['project'], r['run']) == (project, run)
        }


def _run_members(pool: Mapping[str, ProjectPool], project: str, run: str) -> list[PackMember]:
    members = pool[project].members
    if run == 'minus_reservation':
        return list(members)
    return [dataclasses.replace(m, rider_declines=RIDER_RESERVE_MAX_DECLINES) for m in members]


def _fixture_packs(pool: Mapping[str, ProjectPool], project: str, run: str) -> tuple[CandidatePack, ...]:
    hubs = hub_keys(pool[project].members)
    return candidate_packs(_run_members(pool, project, run), hubs=hubs, now=NOW)


@pytest.mark.parametrize('project', PROJECTS)
def test_the_whole_pool_run_reserves_no_member(pool, project):
    assert not any(rider_reserved(m, now=NOW) for m in _run_members(pool, project, 'whole_pool'))


@pytest.mark.parametrize(('project', 'run', 'parent_packs', 'recipe_packs'), _FIXTURE_RUNS)
def test_candidate_packs_reproduces_the_2026_11_04_rerun(pool, project, run, parent_packs, recipe_packs):
    packs = _fixture_packs(pool, project, run)
    assert {(p.generator, frozenset(p.members)): p.key for p in packs} == _expected_packs(project, run)
    assert Counter(p.generator for p in packs) == {'parent': parent_packs, 'recipe': recipe_packs}


@pytest.mark.parametrize('project', PROJECTS)
def test_whole_pool_packs_hold_every_invariant(pool, project):
    by_id = {m.task_id: m for m in pool[project].members}
    packs = _fixture_packs(pool, project, 'whole_pool')
    packed = [task_id for p in packs for task_id in p.members]
    assert len(packed) == len(set(packed))
    for p in packs:
        members = [by_id[task_id] for task_id in p.members]
        assert len(members) >= 2
        assert sum(m.item_count for m in members) <= PACK_MAX_ITEMS
        assert len(p.union_keys) <= PACK_MAX_UNION_KEYS
        assert p.survivor in p.members
        if p.generator == 'parent':
            assert len({(m.parent, m.kind) for m in members}) == 1
