"""The X-Escalation-Project protocol: what a caller sends, what a store honours.

The header-reading arm under a REAL request is proven over HTTP in
``escalation/tests/test_uuid_prefix_guard_registration.py``; reaching
fastmcp's request contextvars from here would pin its internals.
"""

from __future__ import annotations

import pytest

from shared.escalation_project_scope import (
    escalation_project_for,
    escalation_project_headers,
)


def test_the_header_is_spelled_x_escalation_project() -> None:
    assert escalation_project_headers('reify') == {'X-Escalation-Project': 'reify'}


def test_each_call_returns_a_fresh_dict() -> None:
    assert escalation_project_headers('reify') is not escalation_project_headers('reify')


def test_an_empty_project_id_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match=repr('')):
        escalation_project_headers('')


def test_a_store_project_ignores_the_call() -> None:
    project_for = escalation_project_for('reify', ())

    assert project_for({'project_id': 'dark_factory', 'project_root': '/x'}) == 'reify'


def test_a_store_project_with_per_call_projects_is_refused_naming_both() -> None:
    with pytest.raises(ValueError) as refused:
        escalation_project_for('reify', {'dark_factory'})

    assert 'reify' in str(refused.value)
    assert 'dark_factory' in str(refused.value)


def test_a_bare_string_of_projects_is_refused() -> None:
    with pytest.raises(TypeError, match="'reify'"):
        escalation_project_for(None, 'reify')


def test_outside_an_http_request_no_project_is_asserted() -> None:
    assert escalation_project_for(None, {'reify'})({}) is None
