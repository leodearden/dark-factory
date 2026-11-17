"""A SessionRecord builder with fixed defaults for cockpit tests."""

from __future__ import annotations

from orchestrator import session_registry as sr


def make_record(**overrides) -> sr.SessionRecord:
    """Mirrors test_registry_reader.py::_make_record's convention."""
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': sr.Status.RUNNING,
        'title': 'unblock:df#2085 slug',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'start_ts': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)
