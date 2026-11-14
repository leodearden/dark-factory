"""The dashboard payload budget table: its shape, its completeness, its refusals.

``dashboard/src/dashboard/payload_budgets.json`` declares, per REQUEST (a GET
path plus its query), the most wire bytes that request may serve. Two
enforcement points read it: the pre-merge fleet sweep
(``test_payload_budget_sweep.py``) and the live predicate
(``dashboard.payload_check``). This module pins the table itself:

* the shipped table loads;
* it is COMPLETE against the live app's GET routes, in both directions, so a
  new endpoint cannot ship unbudgeted and a deleted one cannot leave a stale
  entry behind;
* a malformed table is REFUSED loudly, naming the field, and never read as an
  empty or partial table, which would compare clean against everything;
* the pure accounting both enforcement points report with: the leaf-field
  breakdown, counted in Starlette's wire bytes, the overage message and the
  slack rule;
* history is APPEND-ONLY against ``merge-base(main, HEAD)``, so every raise or
  lowering is a new entry naming a task and a reason.
"""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _dashboard_routes import dashboard_get_paths
from _payload_budget_history import (
    BaseTable,
    FirstWrite,
    history_violations,
    resolve_base_table,
)
from shared.governed_exceptions import Debt, TicketRef, governed_exceptions
from starlette.responses import JSONResponse

from dashboard.payload_budgets import (
    TABLE_REPO_PATH,
    BudgetTable,
    EndpointBudget,
    FieldBytes,
    PayloadBudgetTableError,
    field_breakdown,
    load_budget_table,
    nodes_at,
    parse_budget_table,
    render_overage,
    slack_violation,
)

PAYLOAD_BUDGET_EXEMPTIONS = governed_exceptions(
    'dashboard.payload_budgets.exempt',
    'A templated GET route under /api/v2/dashboard/ whose response is bounded by one record, '
    'not by population, so it cannot grow organically and carries no payload budget.',
    [exemption.path for exemption in load_budget_table().exempt],
    default=Debt(TicketRef('tkt_0RWZFP07XTSRAT1K10N2GR1Q8D')),
    default_covers=1,
)
"""Every exemption silences the completeness guard below, so each carries an INV-12 disposition."""


@pytest.fixture(scope='module')
def shipped() -> BudgetTable:
    return load_budget_table()


def test_the_shipped_table_loads(shipped: BudgetTable) -> None:
    """parse_budget_table refuses any malformed field (TestRefusals), so loading IS the check."""
    assert shipped.endpoints


class TestUrlSuffix:
    def test_a_bare_path_renders_without_a_query_string(self) -> None:
        entry = parse_budget_table(json.dumps(_minimal_table())).endpoints[0]
        assert entry.url_suffix() == '/api/v2/dashboard/tasks'

    def test_the_query_is_urlencoded_after_the_path(self) -> None:
        raw = _minimal_table()
        raw['endpoints'][0]['query'] = {'terminal': 'dark factory'}
        entry = parse_budget_table(json.dumps(raw)).endpoints[0]
        assert entry.url_suffix() == '/api/v2/dashboard/tasks?terminal=dark+factory'


class TestCompleteness:
    """Entry paths plus exempt paths are EXACTLY the app's GET routes."""

    def test_the_table_covers_every_route_and_nothing_else(
        self, shipped: BudgetTable,
    ) -> None:
        routes = dashboard_get_paths()
        live = set(routes.literal) | set(routes.templated)
        declared = {e.path for e in shipped.endpoints} | {x.path for x in shipped.exempt}

        missing = sorted(live - declared)
        extra = sorted(declared - live)
        assert not missing and not extra, (
            f'payload_budgets.json is out of step with the app. Routes with no '
            f'budget: {missing}. Budgeted paths with no route: {extra}. Remedy: '
            'add an entry with a MEASURED budget and a reason; or, for a '
            'templated route only, an exemption with a reason; or delete the '
            'entry together with its route.'
        )

    def test_entries_and_exemptions_are_disjoint(self, shipped: BudgetTable) -> None:
        both = {e.path for e in shipped.endpoints} & {x.path for x in shipped.exempt}
        assert both == set()

    def test_only_a_templated_route_may_be_exempt(self, shipped: BudgetTable) -> None:
        templated = set(dashboard_get_paths().templated)
        literal_exempt = sorted({x.path for x in shipped.exempt} - templated)
        assert literal_exempt == [], (
            f'{literal_exempt} are exempted from a payload budget but are literal '
            'routes. A literal route serves a population and can grow without '
            'anyone choosing it; give it a measured budget instead.'
        )


def _minimal_table() -> dict[str, Any]:
    return {
        '_README': 'test table',
        'schema_version': 1,
        'endpoints': [
            {
                'path': '/api/v2/dashboard/tasks',
                'query': {},
                'data_keyed_maps': [['TASKS_SNAPSHOT']],
                'history': [
                    {'budget_bytes': 1000, 'task': '5824', 'reason': 'measured 800'},
                ],
            },
        ],
        'exempt': [
            {'path': '/api/v2/dashboard/task/{project}/T-{task_id}', 'reason': 'one task'},
        ],
    }


def _mutated(mutate: Callable[[dict[str, Any]], object]) -> str:
    raw = copy.deepcopy(_minimal_table())
    mutate(raw)
    return json.dumps(raw)


def _entry(raw: dict[str, Any]) -> dict[str, Any]:
    return raw['endpoints'][0]


def _change(raw: dict[str, Any]) -> dict[str, Any]:
    return _entry(raw)['history'][0]


def _duplicate_first_entry(raw: dict[str, Any]) -> None:
    raw['endpoints'].append(copy.deepcopy(_entry(raw)))


def _exempt_a_budgeted_path(raw: dict[str, Any]) -> None:
    raw['exempt'].append({'path': '/api/v2/dashboard/tasks', 'reason': 'both'})


_REFUSALS: list[tuple[str, str, str]] = [
    ('unparseable-json', '{"schema_version": 1,', 'JSON'),
    ('top-level-not-object', '[1, 2]', 'top level'),
    ('schema-missing', _mutated(lambda r: r.pop('schema_version')), 'schema_version'),
    ('schema-wrong', _mutated(lambda r: r.update(schema_version=2)), 'schema_version'),
    ('schema-bool', _mutated(lambda r: r.update(schema_version=True)), 'schema_version'),
    ('unknown-top-level-key', _mutated(lambda r: r.update(budgets=[])), 'budgets'),
    ('endpoints-missing', _mutated(lambda r: r.pop('endpoints')), 'endpoints'),
    ('endpoints-not-list', _mutated(lambda r: r.update(endpoints={})), 'endpoints'),
    ('endpoints-empty', _mutated(lambda r: r.update(endpoints=[])), 'endpoints'),
    ('duplicate-request', _mutated(_duplicate_first_entry), 'duplicate'),
    (
        'path-outside-dashboard-api',
        _mutated(lambda r: _entry(r).update(path='/healthz')),
        'endpoints[0].path',
    ),
    ('query-not-dict', _mutated(lambda r: _entry(r).update(query=[])), 'endpoints[0].query'),
    (
        'query-value-not-str',
        _mutated(lambda r: _entry(r).update(query={'window': 90})),
        'endpoints[0].query',
    ),
    (
        'data-keyed-maps-not-list',
        _mutated(lambda r: _entry(r).update(data_keyed_maps='TASKS_SNAPSHOT')),
        'endpoints[0].data_keyed_maps',
    ),
    (
        'data-keyed-map-path-empty',
        _mutated(lambda r: _entry(r).update(data_keyed_maps=[[]])),
        'endpoints[0].data_keyed_maps',
    ),
    (
        'data-keyed-map-segment-blank',
        _mutated(lambda r: _entry(r).update(data_keyed_maps=[['TASKS', ' ']])),
        'endpoints[0].data_keyed_maps',
    ),
    (
        'history-empty',
        _mutated(lambda r: _entry(r).update(history=[])),
        'endpoints[0].history',
    ),
    (
        'budget-zero',
        _mutated(lambda r: _change(r).update(budget_bytes=0)),
        'endpoints[0].history[0].budget_bytes',
    ),
    (
        'budget-negative',
        _mutated(lambda r: _change(r).update(budget_bytes=-5)),
        'endpoints[0].history[0].budget_bytes',
    ),
    (
        'budget-float',
        _mutated(lambda r: _change(r).update(budget_bytes=1000.0)),
        'endpoints[0].history[0].budget_bytes',
    ),
    (
        'budget-true',
        _mutated(lambda r: _change(r).update(budget_bytes=True)),
        'endpoints[0].history[0].budget_bytes',
    ),
    (
        'task-blank',
        _mutated(lambda r: _change(r).update(task='  ')),
        'endpoints[0].history[0].task',
    ),
    (
        'reason-blank',
        _mutated(lambda r: _change(r).update(reason='')),
        'endpoints[0].history[0].reason',
    ),
    (
        'exempt-missing-reason',
        _mutated(lambda r: r['exempt'][0].pop('reason')),
        'exempt[0].reason',
    ),
    ('path-both-budgeted-and-exempt', _mutated(_exempt_a_budgeted_path), 'exempt'),
]


class TestRefusals:
    """Every malformed table is refused, naming the source and the field."""

    def test_the_minimal_table_itself_parses(self) -> None:
        table = parse_budget_table(json.dumps(_minimal_table()), source='<inline>')
        assert len(table.endpoints) == 1 and len(table.exempt) == 1

    @pytest.mark.parametrize(
        ('text', 'field'),
        [pytest.param(text, field, id=case) for case, text, field in _REFUSALS],
    )
    def test_refuses_naming_source_and_field(self, text: str, field: str) -> None:
        with pytest.raises(PayloadBudgetTableError) as refused:
            parse_budget_table(text, source='budget-table-under-test')

        message = str(refused.value)
        assert 'budget-table-under-test' in message
        assert field in message


def _compact(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def _leaf(key: str, value: object) -> int:
    return _compact(key) + 1 + _compact(value)


def _as_dict(breakdown: tuple[FieldBytes, ...]) -> dict[str, int]:
    return {field.path: field.bytes for field in breakdown}


class TestFieldBreakdown:
    """Bytes are attributed to LEAF fields, with data-keyed maps collapsed."""

    def test_list_indices_collapse_and_rows_sum_into_one_leaf(self) -> None:
        payload = {'SCHEDULER': {'rows': [{'title': 'a'}, {'title': 'bcd'}]}}
        assert _as_dict(field_breakdown(payload)) == {
            'SCHEDULER.rows[].title': _leaf('title', 'a') + _leaf('title', 'bcd'),
        }

    def test_lists_of_dicts_recurse(self) -> None:
        payload = {'A': {'rows': [{'deps': [{'title': 'x'}, {'title': 'yz'}]}]}}
        assert _as_dict(field_breakdown(payload)) == {
            'A.rows[].deps[].title': _leaf('title', 'x') + _leaf('title', 'yz'),
        }

    def test_a_list_of_scalars_is_one_leaf_on_its_key(self) -> None:
        lock_set = ['a/b.py', 'c/d.py']
        payload = {'SCHEDULER': {'rows': [{'lock_set': lock_set}]}}
        assert _as_dict(field_breakdown(payload)) == {
            'SCHEDULER.rows[].lock_set': _leaf('lock_set', lock_set),
        }

    def test_nested_dicts_recurse_and_top_level_scalars_are_leaves(self) -> None:
        payload = {'TASKS_OFFLINE': False, 'WINDOW': {'days': 90, 'served': '90d'}}
        assert _as_dict(field_breakdown(payload)) == {
            'TASKS_OFFLINE': _leaf('TASKS_OFFLINE', False),
            'WINDOW.days': _leaf('days', 90),
            'WINDOW.served': _leaf('served', '90d'),
        }

    def test_empty_containers_are_leaves(self) -> None:
        payload = {'A': {'rows': [], 'meta': {}}}
        assert _as_dict(field_breakdown(payload)) == {
            'A.rows': _leaf('rows', []),
            'A.meta': _leaf('meta', {}),
        }

    def test_a_declared_data_keyed_map_collapses_its_keys(self) -> None:
        payload = {
            'TASKS_SNAPSHOT': {
                'dark-factory': {'rows': {'value': [{'title': 'one'}]}},
                'reify': {'rows': {'value': [{'title': 'two'}, {'title': 'three'}]}},
            },
        }
        collapsed = _as_dict(field_breakdown(payload, (('TASKS_SNAPSHOT',),)))
        assert collapsed == {
            'TASKS_SNAPSHOT.{}.rows.value[].title': (
                _leaf('title', 'one') + _leaf('title', 'two') + _leaf('title', 'three')
            ),
            'TASKS_SNAPSHOT.{}(keys)': _compact('dark-factory') + 1 + _compact('reify') + 1,
        }

    def test_a_collapsed_maps_keys_are_charged_once_beside_its_container_values(self) -> None:
        """A map grown by its keys (longer ids, more of them) shows in the breakdown."""
        payload = {'SCHEDULER': {'events_by_task': {
            '4866': [{'kind': 'claimed'}, {'kind': 'done'}],
            '12345': {'kind': 'claimed'},
        }}}
        collapsed = _as_dict(field_breakdown(payload, (('SCHEDULER', 'events_by_task'),)))
        assert collapsed == {
            'SCHEDULER.events_by_task.{}[].kind': _leaf('kind', 'claimed') + _leaf('kind', 'done'),
            'SCHEDULER.events_by_task.{}.kind': _leaf('kind', 'claimed'),
            'SCHEDULER.events_by_task.{}(keys)': _compact('4866') + 1 + _compact('12345') + 1,
        }

    def test_no_keys_leaf_unless_a_collapsed_map_holds_containers(self) -> None:
        payload = {'WINDOW': {'days': 90}}
        assert _as_dict(field_breakdown(payload, (('WINDOW',),))) == {
            'WINDOW.{}': _leaf('days', 90),
        }
        assert _as_dict(field_breakdown(payload)) == {'WINDOW.days': _leaf('days', 90)}

    def test_an_undeclared_data_keyed_map_stays_split(self) -> None:
        payload = {
            'TASKS_SNAPSHOT': {
                'dark-factory': {'rows': {'value': [{'title': 'one'}]}},
                'reify': {'rows': {'value': [{'title': 'two'}]}},
            },
        }
        assert set(_as_dict(field_breakdown(payload))) == {
            'TASKS_SNAPSHOT.dark-factory.rows.value[].title',
            'TASKS_SNAPSHOT.reify.rows.value[].title',
        }

    def test_a_map_under_a_list_collapses_when_declared_with_list_descent(self) -> None:
        payload = {'A': {'per_project': [{'daily': {'2026-10-01': 1, '2026-10-02': 22}}]}}
        declared = (('A', 'per_project', '[]', 'daily'),)
        assert _as_dict(field_breakdown(payload, declared)) == {
            'A.per_project[].daily.{}': _leaf('2026-10-01', 1) + _leaf('2026-10-02', 22),
        }

    def test_dicts_sharing_a_key_set_are_not_inferred_as_data_keyed(self) -> None:
        """ESCALATIONS.views holds two NAMED Datum fields with identical keys."""
        payload = {'views': {'a': {'value': 1, 'state': 'x'}, 'b': {'value': 2, 'state': 'y'}}}
        assert set(_as_dict(field_breakdown(payload))) == {
            'views.a.value', 'views.a.state', 'views.b.value', 'views.b.state',
        }

    def test_non_ascii_is_counted_in_encoded_bytes(self) -> None:
        payload = {'rows': [{'title': 'déjà vu — ✓'}]}
        assert _as_dict(field_breakdown(payload)) == {
            'rows[].title': _leaf('title', 'déjà vu — ✓'),
        }
        assert _leaf('title', 'déjà vu — ✓') > len('"title":"déjà vu — ✓"')

    def test_a_flat_payloads_leaves_are_its_starlette_body_less_braces_and_commas(self) -> None:
        sample = {
            'title': 'déjà vu — ✓ 日本',
            'n': [1, 2.5, None, True],
            'empty': {},
            'list': [],
        }
        leaves = sum(field.bytes for field in field_breakdown(sample))
        braces_and_commas = 2 + len(sample) - 1
        assert leaves == len(JSONResponse(sample).body) - braces_and_commas

    def test_sorted_by_bytes_descending_then_path(self) -> None:
        payload = {'b': 'xx', 'a': 'xx', 'c': 'xxxxxxxx'}
        assert [field.path for field in field_breakdown(payload)] == ['c', 'a', 'b']

    def test_returns_frozen_field_bytes(self) -> None:
        breakdown = field_breakdown({'a': 1})
        assert isinstance(breakdown, tuple)
        assert breakdown == (FieldBytes(path='a', bytes=_leaf('a', 1)),)


class TestNodesAt:
    """A declared key path resolves in the grammar the breakdown collapses by."""

    def test_named_segments_descend_by_key(self) -> None:
        payload = {'A': {'b': {'c': 1}}}
        assert list(nodes_at(payload, ('A', 'b'))) == [{'c': 1}]

    def test_list_descent_reaches_every_item(self) -> None:
        payload = {'A': {'per_project': [{'daily': {'x': 1}}, {'daily': {'y': 2}}]}}
        assert list(nodes_at(payload, ('A', 'per_project', '[]', 'daily'))) == [{'x': 1}, {'y': 2}]

    def test_a_collapsed_map_reaches_every_value(self) -> None:
        payload = {'B': {'reify': {'labels': {'d1': 1}}, 'dark-factory': {'labels': {'d2': 2}}}}
        assert list(nodes_at(payload, ('B', '{}', 'labels'))) == [{'d1': 1}, {'d2': 2}]

    @pytest.mark.parametrize('key_path', [('missing',), ('A', '[]'), ('A', 'b', 'c', 'd')])
    def test_a_path_that_misses_reaches_nothing(self, key_path: tuple[str, ...]) -> None:
        assert list(nodes_at({'A': {'b': {'c': 1}}}, key_path)) == []


def _entry_for_overage() -> EndpointBudget:
    raw = _minimal_table()
    raw['endpoints'][0]['query'] = {'terminal': 'dark-factory'}
    raw['endpoints'][0]['history'].append(
        {'budget_bytes': 1200, 'task': '9999', 'reason': 'raised after the 2026-10 growth'},
    )
    return parse_budget_table(json.dumps(raw)).endpoints[0]


class TestRenderOverage:
    def test_names_request_numbers_fields_history_and_remedy(self) -> None:
        breakdown = tuple(
            FieldBytes(path=f'TASKS_SNAPSHOT.{{}}.rows.value[].f{i}', bytes=200 - i)
            for i in range(10)
        )
        message = render_overage(_entry_for_overage(), 1500, breakdown, top_n=8)

        assert '/api/v2/dashboard/tasks?terminal=dark-factory' in message
        assert '1,200' in message and '1,500' in message and '1.25' in message
        for field in breakdown[:8]:
            assert field.path in message
            assert f'{field.bytes:,}' in message
            assert f'{100 * field.bytes / 1500:.1f}%' in message
        assert breakdown[8].path not in message and breakdown[9].path not in message
        assert '9999' in message and 'raised after the 2026-10 growth' in message
        assert 'shrink' in message
        assert 'APPEND' in message
        assert 'dashboard/src/dashboard/payload_budgets.json' in message


class TestSlackViolation:
    @pytest.mark.parametrize(
        ('budget', 'measured', 'violates'),
        [
            pytest.param(5_500_000, 2_000_000, True, id='a-large-shrink'),
            pytest.param(5_500_000, 4_374_910, False, id='calibrated'),
            pytest.param(19_000, 1_749, False, id='under-the-floor'),
            pytest.param(5_500_000, 2, True, id='an-empty-payload'),
        ],
    )
    def test_slack_is_budget_beyond_ratio_and_floor(
        self, budget: int, measured: int, violates: bool,
    ) -> None:
        assert slack_violation(budget, measured) is violates


_TASKS = '/api/v2/dashboard/tasks'
_SCHEDULER = '/api/v2/dashboard/scheduler'
_PROSE = '/api/v2/dashboard/task/{project}/T-{task_id}'

_Change = tuple[int, str, str]


def _history_table(
    histories: dict[tuple[str, tuple[tuple[str, str], ...]], list[_Change]],
    exempt: tuple[str, ...] = (),
) -> BudgetTable:
    """A parsed table holding exactly *histories*, keyed by (path, query pairs)."""
    raw = {
        'schema_version': 1,
        'endpoints': [
            {
                'path': path,
                'query': dict(query),
                'data_keyed_maps': [],
                'history': [
                    {'budget_bytes': size, 'task': task, 'reason': reason}
                    for size, task, reason in changes
                ],
            }
            for (path, query), changes in histories.items()
        ],
        'exempt': [{'path': path, 'reason': 'one task'} for path in exempt],
    }
    return parse_budget_table(json.dumps(raw))


_SEED: _Change = (5_500_000, '5824', 'measured 4,351,859 on 2026-10-15')
_RAISE: _Change = (6_400_000, '6001', 'rows grew to 5,800; shrinking meta_files is task 6002')
_LOWER: _Change = (3_000_000, '6003', 'meta_files moved to the prose route')

_BARE_TASKS = (_TASKS, ())
_CENSUS = (_TASKS, (('projection', 'census'),))


class TestHistoryViolations:
    """The DECISION layer: head's history must extend base's, request by request."""

    def test_identical_tables_are_clean(self) -> None:
        table = _history_table({_BARE_TASKS: [_SEED]})
        assert history_violations(table, table) == []

    def test_an_appended_raise_is_clean(self) -> None:
        base = _history_table({_BARE_TASKS: [_SEED]})
        head = _history_table({_BARE_TASKS: [_SEED, _RAISE]})
        assert history_violations(base, head) == []

    def test_an_appended_lowering_is_clean(self) -> None:
        base = _history_table({_BARE_TASKS: [_SEED]})
        head = _history_table({_BARE_TASKS: [_SEED, _LOWER]})
        assert history_violations(base, head) == []

    @pytest.mark.parametrize(
        ('edited', 'old', 'new'),
        [
            pytest.param((9_000_000, '5824', _SEED[2]), '5500000', '9000000', id='budget'),
            pytest.param((_SEED[0], '6001', _SEED[2]), "'5824'", "'6001'", id='task'),
            pytest.param((_SEED[0], '5824', 'rewritten'), _SEED[2], 'rewritten', id='reason'),
        ],
    )
    def test_an_in_place_edit_is_one_violation_naming_the_change(
        self, edited: _Change, old: str, new: str,
    ) -> None:
        base = _history_table({_BARE_TASKS: [_SEED, _RAISE]})
        head = _history_table({_BARE_TASKS: [edited, _RAISE]})

        violations = history_violations(base, head)

        assert len(violations) == 1, violations
        assert _TASKS in violations[0]
        assert 'history[0]' in violations[0]
        assert old in violations[0] and new in violations[0]

    def test_a_deleted_entry_is_a_violation(self) -> None:
        base = _history_table({_BARE_TASKS: [_SEED, _RAISE]})
        head = _history_table({_BARE_TASKS: [_SEED]})
        violations = history_violations(base, head)
        assert len(violations) == 1 and 'history[1]' in violations[0], violations

    def test_a_reordered_history_is_a_violation(self) -> None:
        base = _history_table({_BARE_TASKS: [_SEED, _RAISE]})
        head = _history_table({_BARE_TASKS: [_RAISE, _SEED]})
        assert len(history_violations(base, head)) == 1

    def test_a_request_added_at_head_is_clean(self) -> None:
        base = _history_table({_BARE_TASKS: [_SEED]})
        head = _history_table({_BARE_TASKS: [_SEED], (_SCHEDULER, ()): [_SEED]})
        assert history_violations(base, head) == []

    def test_a_request_removed_at_head_is_left_to_completeness(self) -> None:
        base = _history_table({_BARE_TASKS: [_SEED], (_SCHEDULER, ()): [_SEED]})
        head = _history_table({_BARE_TASKS: [_SEED]})
        assert history_violations(base, head) == []

    def test_exemptions_added_or_removed_are_clean(self) -> None:
        with_exemption = _history_table({_BARE_TASKS: [_SEED]}, exempt=(_PROSE,))
        without = _history_table({_BARE_TASKS: [_SEED]})
        assert history_violations(with_exemption, without) == []
        assert history_violations(without, with_exemption) == []

    def test_the_same_path_with_another_query_is_another_request(self) -> None:
        base = _history_table({_BARE_TASKS: [_SEED]})
        head = _history_table({_CENSUS: [_LOWER]})
        assert history_violations(base, head) == []


def _init_repo(path: Path) -> Path:
    """``git init`` a fresh repo at *path* with one commit on ``main``.

    Copied from test_cache_buster_freshness.py::_init_repo rather than
    imported, for the reason that module states: a test module's helpers are
    not imported across test modules. The suite-wide git ceiling keeps it
    inside the pytest basetemp.
    """
    path.mkdir(parents=True, exist_ok=True)
    _git(path, 'init', '-b', 'main')
    _git(path, 'config', 'user.email', 'test@test.com')
    _git(path, 'config', 'user.name', 'Test')
    (path / 'README.md').write_text('# sentinel\n')
    _git(path, 'add', '-A')
    _git(path, 'commit', '-m', 'initial')
    return path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout


def _commit_table(repo: Path, histories: list[_Change], message: str) -> None:
    raw = {
        'schema_version': 1,
        'endpoints': [{
            'path': _TASKS,
            'query': {},
            'data_keyed_maps': [],
            'history': [
                {'budget_bytes': size, 'task': task, 'reason': reason}
                for size, task, reason in histories
            ],
        }],
        'exempt': [],
    }
    table = repo / TABLE_REPO_PATH
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(raw))
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-m', message)


class TestResolveBaseTable:
    """The GIT layer, against throwaway repos built to exhibit each answer."""

    def test_no_main_ref_is_unmeasurable(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path / 'repo')
        _git(repo, 'branch', '-m', 'main', 'trunk')
        assert resolve_base_table(repo) is None

    def test_a_table_absent_at_the_base_is_a_first_write(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path / 'repo')
        _git(repo, 'checkout', '-b', 'work')
        _commit_table(repo, [_SEED], 'first write of the table')

        base = resolve_base_table(repo)

        assert isinstance(base, FirstWrite), base
        assert base.base_ref == 'main'
        assert base.base_commit == _git(repo, 'rev-parse', 'main').strip()

    def test_the_table_is_read_at_the_merge_base_not_at_mains_tip(
        self, tmp_path: Path,
    ) -> None:
        repo = _init_repo(tmp_path / 'repo')
        _commit_table(repo, [_SEED], 'seed the table')
        branch_point = _git(repo, 'rev-parse', 'HEAD').strip()
        _git(repo, 'checkout', '-b', 'work')
        _git(repo, 'checkout', 'main')
        _commit_table(repo, [_SEED, _RAISE], 'main raises after the branch point')
        _git(repo, 'checkout', 'work')

        base = resolve_base_table(repo)

        assert isinstance(base, BaseTable), base
        assert base.base_commit == branch_point
        assert [c.budget_bytes for c in base.table.endpoints[0].history] == [_SEED[0]]

    def test_the_second_candidate_ref_is_tried(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path / 'repo')
        _commit_table(repo, [_SEED], 'seed the table')
        _git(repo, 'checkout', '-b', 'work')
        _git(repo, 'update-ref', 'refs/remotes/origin/main', 'main')
        _git(repo, 'branch', '-D', 'main')

        base = resolve_base_table(repo)

        assert isinstance(base, BaseTable) and base.base_ref == 'origin/main', base

    def test_a_git_failure_after_merge_base_raises_with_stderr(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path / 'repo')
        _commit_table(repo, [_SEED], 'seed the table')
        _git(repo, 'checkout', '-b', 'work')
        blob = _git(repo, 'rev-parse', f'main:{TABLE_REPO_PATH}').strip()
        (repo / '.git' / 'objects' / blob[:2] / blob[2:]).unlink()

        with pytest.raises(RuntimeError) as failed:
            resolve_base_table(repo)

        assert 'fatal:' in str(failed.value), 'the refusal must carry git\'s own stderr'


_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.reads_git_history
def test_the_shipped_history_only_appends_to_the_merge_base() -> None:
    """THE GUARD: no recorded budget entry was edited, reordered or deleted."""
    base = resolve_base_table(_REPO_ROOT)
    if base is None:
        pytest.skip(
            f'neither main nor origin/main resolves in {_REPO_ROOT}, so there is '
            'no merge base to compare the budget history against'
        )
    if isinstance(base, FirstWrite):
        pytest.skip(
            f'{TABLE_REPO_PATH} does not exist at merge base {base.base_commit[:12]} '
            f'({base.base_ref}); this branch is its first write, so there is no '
            'recorded history to protect yet'
        )

    violations = history_violations(base.table, load_budget_table())

    assert violations == [], (
        f'{TABLE_REPO_PATH} rewrites budget history recorded at merge base '
        f'{base.base_commit[:12]} ({base.base_ref}):\n  ' + '\n  '.join(violations)
        + '\nHistory is APPEND-ONLY: a budget is the last entry, so every raise or '
        'lowering is a NEW entry naming its task and reason. Restore the recorded '
        'entries exactly and append a new one instead.'
    )
