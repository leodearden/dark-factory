"""The dashboard payload budget table: its shape, its completeness, its refusals.

``dashboard/src/dashboard/payload_budgets.json`` declares, per REQUEST (a GET
path plus its query), the most wire bytes that request may serve. Two
enforcement points read it: the pre-merge fleet sweep
(``test_payload_budget_sweep.py``) and the live predicate
(``dashboard.payload_check``). This module pins the table itself:

* the shipped table loads into well-typed, read-only values;
* it is COMPLETE against the live app's GET routes, in both directions, so a
  new endpoint cannot ship unbudgeted and a deleted one cannot leave a stale
  entry behind;
* a malformed table is REFUSED loudly, naming the field, and never read as an
  empty or partial table, which would compare clean against everything;
* the pure accounting both enforcement points report with: wire bytes, the
  leaf-field breakdown, the overage message and the slack rule.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import pytest
from _dashboard_routes import dashboard_get_paths
from starlette.responses import JSONResponse

from dashboard.payload_budgets import (
    MAX_SLACK_RATIO,
    SLACK_FLOOR_BYTES,
    BudgetChange,
    BudgetTable,
    EndpointBudget,
    Exemption,
    FieldBytes,
    PayloadBudgetTableError,
    field_breakdown,
    load_budget_table,
    parse_budget_table,
    payload_bytes,
    render_overage,
    slack_violation,
)

_PREFIX = '/api/v2/dashboard/'


@pytest.fixture(scope='module')
def shipped() -> BudgetTable:
    return load_budget_table()


class TestShippedTable:
    def test_loads_endpoints_and_exemptions(self, shipped: BudgetTable) -> None:
        assert isinstance(shipped, BudgetTable)
        assert isinstance(shipped.endpoints, tuple)
        assert shipped.endpoints, 'the shipped table declares no endpoint at all'
        assert all(isinstance(e, EndpointBudget) for e in shipped.endpoints)
        assert isinstance(shipped.exempt, tuple)
        assert all(isinstance(x, Exemption) for x in shipped.exempt)

    def test_every_path_is_under_the_dashboard_api(self, shipped: BudgetTable) -> None:
        outside = [e.path for e in shipped.endpoints if not e.path.startswith(_PREFIX)]
        assert outside == []

    def test_each_request_appears_once(self, shipped: BudgetTable) -> None:
        keys = [e.request_key for e in shipped.endpoints]
        assert len(keys) == len(set(keys))

    def test_query_is_a_read_only_str_mapping(self, shipped: BudgetTable) -> None:
        for entry in shipped.endpoints:
            assert isinstance(entry.query, MappingProxyType), entry.url_suffix()
            assert all(
                isinstance(k, str) and isinstance(v, str) for k, v in entry.query.items()
            ), entry.url_suffix()

    def test_data_keyed_maps_are_tuples_of_non_blank_segments(
        self, shipped: BudgetTable,
    ) -> None:
        for entry in shipped.endpoints:
            assert isinstance(entry.data_keyed_maps, tuple), entry.url_suffix()
            for key_path in entry.data_keyed_maps:
                assert isinstance(key_path, tuple) and key_path, entry.url_suffix()
                assert all(isinstance(s, str) and s.strip() for s in key_path)

    def test_history_is_a_non_empty_tuple_of_reasoned_changes(
        self, shipped: BudgetTable,
    ) -> None:
        for entry in shipped.endpoints:
            assert isinstance(entry.history, tuple) and entry.history, entry.url_suffix()
            for change in entry.history:
                assert isinstance(change, BudgetChange)
                assert type(change.budget_bytes) is int and change.budget_bytes > 0
                assert change.task.strip() and change.reason.strip()

    def test_the_budget_is_the_last_history_entry(self, shipped: BudgetTable) -> None:
        for entry in shipped.endpoints:
            assert entry.budget_bytes == entry.history[-1].budget_bytes

    def test_every_exemption_carries_a_reason(self, shipped: BudgetTable) -> None:
        assert all(x.reason.strip() for x in shipped.exempt)


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
        }

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

    def test_sorted_by_bytes_descending_then_path(self) -> None:
        payload = {'b': 'xx', 'a': 'xx', 'c': 'xxxxxxxx'}
        assert [field.path for field in field_breakdown(payload)] == ['c', 'a', 'b']

    def test_returns_frozen_field_bytes(self) -> None:
        breakdown = field_breakdown({'a': 1})
        assert isinstance(breakdown, tuple)
        assert breakdown == (FieldBytes(path='a', bytes=_leaf('a', 1)),)


class TestPayloadBytes:
    def test_matches_the_starlette_json_response_body(self) -> None:
        sample = {
            'title': 'déjà vu — ✓ 日本',
            'n': [1, 2.5, None, True],
            'nested': {'empty': {}, 'list': []},
        }
        assert payload_bytes(sample) == len(JSONResponse(sample).body)


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
    def test_the_constants_are_the_documented_ones(self) -> None:
        assert MAX_SLACK_RATIO == 2.0
        assert SLACK_FLOOR_BYTES == 32_768

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
