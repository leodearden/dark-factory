"""The pre-merge payload budget sweep: every budgeted request, through the real app.

Each request in ``payload_budgets.json`` is served by the REAL routes and
shapers over a synthetic fleet calibrated to the live population
(``_payload_fleet.py``), faked only at the MCP, database and file seams. So a
field added to a shared row shaper shows up here as bytes, before it merges.

Four assertions per request:

* it answers 200;
* its wire bytes are within budget, failing with the largest leaf fields;
* the budget is not SLACK against it, which catches a fleet that stopped
  reaching the endpoint's data and a real shrink nobody recorded;
* every declared data-keyed map is present, so a declaration cannot rot.

The fleet is expensive to build, so the module runs in one xdist group.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest
from _payload_fleet import open_fleet_client
from fastapi.testclient import TestClient

from dashboard.payload_budgets import (
    EndpointBudget,
    field_breakdown,
    load_budget_table,
    render_overage,
    slack_violation,
)

pytestmark = pytest.mark.xdist_group('payload_fleet')

_TASK_SUBSTRATE_REQUESTS: tuple[tuple[str, Mapping[str, str]], ...] = (
    ('/api/v2/dashboard/tasks', {}),
    ('/api/v2/dashboard/tasks', {'projection': 'census'}),
    ('/api/v2/dashboard/tasks', {'terminal': 'dark-factory'}),
    ('/api/v2/dashboard/scheduler', {}),
    ('/api/v2/dashboard/orchestrators', {}),
)


def _entry(path: str, query: Mapping[str, str]) -> EndpointBudget:
    wanted = (path, tuple(sorted(query.items())))
    for entry in load_budget_table().endpoints:
        if entry.request_key == wanted:
            return entry
    raise LookupError(f'payload_budgets.json has no entry for {path} with query {dict(query)}')


_SWEPT = tuple(_entry(path, query) for path, query in _TASK_SUBSTRATE_REQUESTS)


@pytest.fixture(scope='module')
def fleet_client() -> Iterator[TestClient]:
    with open_fleet_client() as client:
        yield client


def _targets(node: object, key_path: tuple[str, ...]) -> Iterator[object]:
    """Every node *key_path* reaches; ``[]`` descends a list, ``{}`` every map value."""
    if not key_path:
        yield node
        return
    segment, rest = key_path[0], key_path[1:]
    if segment == '[]' and isinstance(node, list):
        children = node
    elif segment == '{}' and isinstance(node, dict):
        children = list(node.values())
    elif isinstance(node, dict) and segment in node:
        children = [node[segment]]
    else:
        children = []
    for child in children:
        yield from _targets(child, rest)


@pytest.mark.parametrize('entry', _SWEPT, ids=[entry.url_suffix() for entry in _SWEPT])
def test_request_is_within_its_budget(
    fleet_client: TestClient, entry: EndpointBudget,
) -> None:
    resp = fleet_client.get(entry.path, params=dict(entry.query))
    assert resp.status_code == 200, f'{entry.url_suffix()} answered {resp.status_code}'

    measured = len(resp.content)
    payload = resp.json()
    assert measured <= entry.budget_bytes, render_overage(
        entry, measured, field_breakdown(payload, entry.data_keyed_maps),
    )

    assert not slack_violation(entry.budget_bytes, measured), (
        f'{entry.url_suffix()} served only {measured:,} bytes over the fleet against a '
        f'budget of {entry.budget_bytes:,}, so the budget is SLACK. Two causes:\n'
        '  FIXTURE ROT: the fleet no longer reaches this endpoint\'s data (a renamed '
        'MCP tool, a moved DB path, a queue no longer discovered); check '
        'dashboard/tests/_payload_fleet.py and _payload_fleet_seed.py.\n'
        '  A REAL SHRINK: the payload got smaller; tighten the budget by APPENDING a '
        'lowered history entry with your task id and the new live measurement.'
    )

    for key_path in entry.data_keyed_maps:
        reached = [node for node in _targets(payload, key_path) if isinstance(node, dict)]
        assert any(reached), (
            f'{entry.url_suffix()} declares the data-keyed map {list(key_path)}, but it '
            'resolves to no non-empty dict in the served payload. The declaration has '
            'rotted (a key was renamed or moved); fix data_keyed_maps in '
            'payload_budgets.json.'
        )
