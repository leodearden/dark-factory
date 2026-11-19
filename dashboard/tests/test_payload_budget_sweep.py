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

A request whose route waits a bounded time on its reads is measured once those
reads have landed, as a polled dashboard serves it
(``dashboard/tests/_payload_fleet.py::get_landed``).

The fleet is expensive to build, so the module runs in one xdist group.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from _payload_fleet import CENSUS, get_landed, open_fleet_client
from _served_reads import forget_served_reads
from fastapi.testclient import TestClient

import dashboard.api.prds
from dashboard.payload_budgets import (
    EndpointBudget,
    field_breakdown,
    load_budget_table,
    nodes_at,
    render_overage,
    slack_violation,
)

pytestmark = pytest.mark.xdist_group('payload_fleet')

_SWEPT: tuple[EndpointBudget, ...] = load_budget_table().endpoints
"""Every budgeted request, so a new table entry (and, through completeness, a new route) is swept."""


@pytest.fixture(scope='module')
def fleet_client() -> Iterator[TestClient]:
    with open_fleet_client() as client:
        yield client


@pytest.mark.parametrize('entry', _SWEPT, ids=[entry.url_suffix() for entry in _SWEPT])
def test_request_is_within_its_budget(
    fleet_client: TestClient, entry: EndpointBudget,
) -> None:
    resp = get_landed(fleet_client, entry)
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
        reached = [node for node in nodes_at(payload, key_path) if isinstance(node, dict)]
        assert any(reached), (
            f'{entry.url_suffix()} declares the data-keyed map {list(key_path)}, but it '
            'resolves to no non-empty dict in the served payload. The declaration has '
            'rotted (a key was renamed or moved); fix data_keyed_maps in '
            'payload_budgets.json.'
        )


def test_a_prds_request_whose_activity_reads_miss_their_wait_is_measured_once_they_land(
    fleet_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.api.prds, 'ACTIVITY_REQUEST_WAIT_SECONDS', 0.0)
    forget_served_reads()
    entry = next(entry for entry in _SWEPT if entry.path == '/api/v2/dashboard/prds')

    resp = get_landed(fleet_client, entry)

    assert resp.status_code == 200
    fresh = {label for label, view in resp.json()['PRD_VIEW'].items() if view['state'] == 'fresh'}
    assert fresh == {root.label for root in CENSUS.roots if root.holds_tasks}
    assert not slack_violation(entry.budget_bytes, len(resp.content))
