"""Unit tests for dashboard.data.tasks._shape_task.

Focus: the field-mapping contract at the MCP→dashboard boundary.
"""

from __future__ import annotations

import pytest

from dashboard.data.tasks import _shape_task


# ---------------------------------------------------------------------------
# updated_at preservation (step-1/step-2)
# ---------------------------------------------------------------------------


def test_shape_task_preserves_updated_at():
    """_shape_task must carry MCP 'updatedAt' through as 'updated_at'."""
    raw = {
        'id': '7',
        'title': 'my task',
        'status': 'done',
        'updatedAt': '2026-05-29T10:00:00+00:00',
        'dependencies': [],
        'metadata': {},
    }
    shaped = _shape_task(raw)
    assert shaped is not None
    assert shaped['updated_at'] == '2026-05-29T10:00:00+00:00'


def test_shape_task_updated_at_none_when_absent():
    """updated_at must be None (not KeyError) when updatedAt is missing."""
    raw = {
        'id': '8',
        'title': 'other task',
        'status': 'pending',
        'dependencies': [],
        'metadata': {},
    }
    shaped = _shape_task(raw)
    assert shaped is not None
    # Must be present in the dict with value None (not missing key)
    assert 'updated_at' in shaped
    assert shaped['updated_at'] is None


def test_shape_task_updated_at_none_when_explicitly_null():
    """updated_at must be None when updatedAt is explicitly None."""
    raw = {
        'id': '9',
        'title': 'null task',
        'status': 'in-progress',
        'updatedAt': None,
        'dependencies': [],
        'metadata': {},
    }
    shaped = _shape_task(raw)
    assert shaped is not None
    assert shaped['updated_at'] is None


# ---------------------------------------------------------------------------
# Existing invariants: id coercion, None on invalid id
# ---------------------------------------------------------------------------


def test_shape_task_coerces_string_id_to_int():
    raw = {'id': '42', 'title': 'x', 'status': 'pending', 'dependencies': []}
    shaped = _shape_task(raw)
    assert shaped is not None
    assert shaped['id'] == 42


def test_shape_task_returns_none_on_missing_id():
    assert _shape_task({'title': 'no id', 'status': 'pending'}) is None


def test_shape_task_returns_none_on_non_numeric_id():
    assert _shape_task({'id': 'abc', 'title': 'bad id', 'status': 'pending'}) is None
