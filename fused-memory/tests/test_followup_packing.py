"""The held-follow-up pack generator: plans/followup-packing-sweep-prd.md §4.1.

Synthetic members pin each rule. The 2026-11-04 pool dump and the a2+c2 CSV the re-run produced
from it (copied verbatim under ``fixtures/``) pin the whole generator: built from every dump
record, ``candidate_packs`` must reproduce that CSV.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from shared.followup_items import MAX_FOLLOWUP_ITEMS

from fused_memory.middleware.followup_packing import (
    HUB_SHARE,
    PACK_MAX_ITEMS,
    PACK_MAX_UNION_KEYS,
    RIDER_RESERVE_DAYS,
    RIDER_RESERVE_MAX_DECLINES,
    PackExclusion,
    PackMember,
    PackRecipe,
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
        'status': 'pending',
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
        status='pending',
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
    ],
)
def test_member_refuses_a_value_outside_its_vocabulary(overrides, field, value):
    with pytest.raises(ValueError, match=field) as refused:
        member(**overrides)
    assert value in str(refused.value)


def test_member_is_frozen():
    frozen: Any = member()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.title = 'renamed'


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
        ({'status': 'in-progress'}, PackExclusion.NOT_HELD),
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
