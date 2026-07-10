"""Tests for cockpit.registry_reader — scan_sessions + in-memory tick diff.

scan_sessions is the cockpit's sole read path over the C1 session registry
(orchestrator.session_registry, PRD §6 G5: consumers import the frozen
contract, never re-derive it). Fail-soft is a hard constraint (PRD §2): a
corrupt record must never abort the whole scan.

build_snapshot/snapshot_changed drive the app's rebuild-only-on-change poll:
a snapshot is keyed on substantive display fields only (never start_ts/age),
so a purely-time-passing tick is a no-op diff.
"""

from __future__ import annotations

from orchestrator import session_registry as sr


def _make_record(**overrides):
    """Build a SessionRecord with sane defaults; overrides tweak individual fields.

    Mirrors orchestrator/tests/test_session_registry.py's _make_record
    helper convention (a fields dict + .update(overrides)).
    """
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


class TestScanSessions:
    def test_returns_all_seeded_records(self, tmp_path):
        from cockpit.registry_reader import scan_sessions

        r1 = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        r2 = _make_record(session_slug='b-2', status=sr.Status.AWAITING_INPUT)
        r3 = _make_record(session_slug='c-3', status=sr.Status.IDLE)
        for r in (r1, r2, r3):
            sr.write_record(r, root=tmp_path)

        result = scan_sessions(tmp_path)

        assert {r.session_slug for r in result} == {'a-1', 'b-2', 'c-3'}
        assert all(isinstance(r, sr.SessionRecord) for r in result)

    def test_skips_corrupt_record_without_aborting_scan(self, tmp_path):
        from cockpit.registry_reader import scan_sessions

        good = _make_record(session_slug='good-1', status=sr.Status.RUNNING)
        sr.write_record(good, root=tmp_path)

        corrupt_dir = sr.sessions_dir(tmp_path) / 'corrupt-2'
        corrupt_dir.mkdir(parents=True)
        (corrupt_dir / 'record.json').write_text('{not valid json')

        result = scan_sessions(tmp_path)

        assert {r.session_slug for r in result} == {'good-1'}

    def test_skips_slug_dir_with_missing_record_file(self, tmp_path):
        """A slug dir with no record.json at all (FileNotFoundError) is skipped too."""
        from cockpit.registry_reader import scan_sessions

        good = _make_record(session_slug='good-1', status=sr.Status.RUNNING)
        sr.write_record(good, root=tmp_path)

        empty_dir = sr.sessions_dir(tmp_path) / 'empty-2'
        empty_dir.mkdir(parents=True)

        result = scan_sessions(tmp_path)

        assert {r.session_slug for r in result} == {'good-1'}

    def test_missing_sessions_dir_returns_empty_list(self, tmp_path):
        from cockpit.registry_reader import scan_sessions

        result = scan_sessions(tmp_path)

        assert result == []
