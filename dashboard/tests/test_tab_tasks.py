"""Source-structure contract tests for the Tasks tab UI (frontend).

Tests parse JSX source as text via Starlette TestClient and assert
structural contracts (presence of External-dependencies section, correct
rendering of dep id + status, status-driven chip class).

Follows the idiom established in test_tab_scheduler.py and test_tab_curator.py.
No JS test runner exists; these are regex/structural assertions only.
Full visual rendering is verified via Playwright/the dashboard read path.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def tab_tasks_jsx_body(_client):
    return _client.get('/static/redux/tab_tasks.jsx').text


# ---------------------------------------------------------------------------
# step-9 / step-10: External dependencies section in TaskDetail
# ---------------------------------------------------------------------------


def test_task_detail_reads_external_deps(tab_tasks_jsx_body):
    """TaskDetail must read task.external_deps (or default to [])."""
    assert re.search(
        r'task\.external_deps',
        tab_tasks_jsx_body,
    ), (
        'TaskDetail must access task.external_deps to render cross-project deps'
    )


def test_task_detail_renders_external_deps_section_label(tab_tasks_jsx_body):
    """TaskDetail must emit a section label matching /External depend/i with a count."""
    assert re.search(
        r'External\s+depend',
        tab_tasks_jsx_body,
        re.IGNORECASE,
    ), (
        'TaskDetail must have an "External dependencies" section label'
    )
    # The count must be dynamic (referencing extDeps or external_deps length)
    assert re.search(
        r'ext[Dd]eps?\.length|external_deps[^\)]*\.length',
        tab_tasks_jsx_body,
    ), (
        'TaskDetail "External dependencies" section label must show a dynamic count '
        'derived from extDeps.length or similar'
    )


def test_task_detail_renders_dep_id_in_external_section(tab_tasks_jsx_body):
    """TaskDetail must render each dep's id inside the External-dependencies section."""
    # The mapping over extDeps must reference d.id
    assert re.search(
        r'ext[Dd]eps?\.map\([^)]*\)|map\(d\s*=>\s*[^)]*d\.id',
        tab_tasks_jsx_body,
    ), (
        'TaskDetail must map over extDeps and render d.id'
    )
    assert re.search(
        r'd\.id',
        tab_tasks_jsx_body,
    ), (
        'TaskDetail external deps chip must render d.id'
    )


def test_task_detail_renders_dep_status_in_external_section(tab_tasks_jsx_body):
    """TaskDetail must render each dep's status in the chip body."""
    assert re.search(
        r'd\.status',
        tab_tasks_jsx_body,
    ), (
        'TaskDetail external deps chip must render d.status'
    )


def test_task_detail_external_section_uses_status_driven_chip_class(tab_tasks_jsx_body):
    """External-dep chips must use a status-driven CSS class (dep-done / dep-pending or similar)."""
    # Must reference 'dep-done' for the resolved/done status
    assert 'dep-done' in tab_tasks_jsx_body, (
        'TaskDetail external deps chip must use dep-done class for done status'
    )
    # Must reference 'dep-pending' or 'dep-unknown' for non-done statuses
    assert re.search(r'dep-pending|dep-unknown', tab_tasks_jsx_body), (
        'TaskDetail external deps chip must use dep-pending or dep-unknown class for non-done status'
    )


def test_task_detail_external_section_hidden_when_empty(tab_tasks_jsx_body):
    """External-dependencies section must be hidden when extDeps is empty (mirrors Module-locks pattern)."""
    # The section must be conditionally rendered: return null / && / ternary guarded by length check
    assert re.search(
        r'ext[Dd]eps?\.length\s*[=><!]|ext[Dd]eps?\s*&&|!ext[Dd]eps?',
        tab_tasks_jsx_body,
    ), (
        'TaskDetail External-dependencies section must be conditionally rendered '
        '(hidden when extDeps is empty, like Module-locks null-when-empty pattern)'
    )
