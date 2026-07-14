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

import os

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


class TestSessionScanner:
    """SessionScanner is the stateful, cache-aware counterpart to scan_sessions
    (registry_reader.py's plain free function) -- see TestScanChangeShortCircuit
    below for the mtime-cache behavior itself. This class only pins parity
    with scan_sessions and correct on-disk-set tracking across calls.
    """

    def test_scan_matches_scan_sessions_for_seeded_dir(self, tmp_path):
        from cockpit.registry_reader import SessionScanner, scan_sessions

        r1 = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        r2 = _make_record(session_slug='b-2', status=sr.Status.AWAITING_INPUT)
        r3 = _make_record(session_slug='c-3', status=sr.Status.IDLE)
        for r in (r1, r2, r3):
            sr.write_record(r, root=tmp_path)

        scanner = SessionScanner(root=tmp_path)
        result = scanner.scan()

        expected = scan_sessions(tmp_path)
        # Both iterate sorted(base.iterdir()), so scan() must mirror
        # scan_sessions()'s ORDER, not just its content -- a dict comparison
        # alone is order-insensitive and would miss a regression that
        # reordered scan()'s output (e.g. iterating an unsorted dict).
        assert [r.session_slug for r in result] == [r.session_slug for r in expected]
        assert {r.session_slug: r for r in result} == {r.session_slug: r for r in expected}

    def test_scan_reflects_slug_added_since_last_scan(self, tmp_path):
        from cockpit.registry_reader import SessionScanner

        r1 = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        sr.write_record(r1, root=tmp_path)

        scanner = SessionScanner(root=tmp_path)
        first = scanner.scan()
        assert {r.session_slug for r in first} == {'a-1'}

        r2 = _make_record(session_slug='b-2', status=sr.Status.AWAITING_INPUT)
        sr.write_record(r2, root=tmp_path)

        second = scanner.scan()
        assert {r.session_slug for r in second} == {'a-1', 'b-2'}

    def test_scan_drops_slug_removed_since_last_scan(self, tmp_path):
        """The second scan() reflects the CURRENT on-disk set, not a stale
        union of everything ever seen."""
        import shutil

        from cockpit.registry_reader import SessionScanner

        r1 = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        r2 = _make_record(session_slug='b-2', status=sr.Status.AWAITING_INPUT)
        for r in (r1, r2):
            sr.write_record(r, root=tmp_path)

        scanner = SessionScanner(root=tmp_path)
        first = scanner.scan()
        assert {r.session_slug for r in first} == {'a-1', 'b-2'}

        shutil.rmtree(sr.sessions_dir(tmp_path) / 'a-1')

        second = scanner.scan()
        assert {r.session_slug for r in second} == {'b-2'}


class TestScanChangeShortCircuit:
    """Proves SessionScanner's mtime cache: a scan() call must not re-parse
    (call session_registry.read_record for) a slug whose record.json mtime
    is unchanged since the previous scan() -- this is the ~4.5s-at-10k-
    sessions cost this task exists to eliminate (see the module docstring).
    """

    _FIXED_NS = 1_700_000_000_000_000_000

    def test_unchanged_mtime_skips_reparse(self, tmp_path, monkeypatch):
        from cockpit import registry_reader
        from cockpit.registry_reader import SessionScanner

        record = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        sr.write_record(record, root=tmp_path)
        path = sr.record_path_for_slug('a-1', root=tmp_path)
        os.utime(path, ns=(self._FIXED_NS, self._FIXED_NS))

        call_count = 0
        original_read_record = registry_reader.session_registry.read_record

        def counting_read_record(slug, root=None):
            nonlocal call_count
            call_count += 1
            return original_read_record(slug, root=root)

        monkeypatch.setattr(registry_reader.session_registry, 'read_record', counting_read_record)

        scanner = SessionScanner(root=tmp_path)
        first = scanner.scan()
        assert call_count == 1
        assert {r.session_slug for r in first} == {'a-1'}

        # record.json's mtime is untouched -- the second scan must reuse the
        # cached SessionRecord rather than calling read_record again.
        second = scanner.scan()
        assert call_count == 1
        assert {r.session_slug: r for r in second} == {r.session_slug: r for r in first}

    def test_changed_mtime_reparses_and_reflects_new_value(self, tmp_path):
        """A genuine rewrite (write_record/update_status, which always goes
        through os.replace and so always bumps record.json's mtime) must be
        picked up by the very next scan() -- the cache never masks a real
        change."""
        from cockpit.registry_reader import SessionScanner

        record = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        sr.write_record(record, root=tmp_path)

        scanner = SessionScanner(root=tmp_path)
        first = scanner.scan()
        assert len(first) == 1
        assert first[0].status == sr.Status.RUNNING

        sr.update_status('a-1', root=tmp_path, status=sr.Status.IDLE)

        second = scanner.scan()
        assert len(second) == 1
        assert second[0].status == sr.Status.IDLE

    def test_same_mtime_different_size_rewrite_is_not_masked_by_cache(self, tmp_path):
        """The cache is keyed on (st_mtime_ns, st_size), not st_mtime_ns
        alone: a rewrite that coincidentally lands on the exact same mtime
        (plausible on filesystems whose real mtime resolution is coarser
        than the nanoseconds st_mtime_ns reports) must still be detected
        when it also changes the record's serialized length, rather than
        being masked as a false cache hit.
        """
        from cockpit.registry_reader import SessionScanner

        record = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        sr.write_record(record, root=tmp_path)
        path = sr.record_path_for_slug('a-1', root=tmp_path)
        os.utime(path, ns=(self._FIXED_NS, self._FIXED_NS))

        scanner = SessionScanner(root=tmp_path)
        first = scanner.scan()
        assert first[0].status == sr.Status.RUNNING

        # 'running' (7 chars) -> 'awaiting-input' (14 chars) changes
        # record.json's byte length; re-pin the mtime back to the exact
        # value the cache already holds to force the adversarial collision.
        sr.update_status('a-1', root=tmp_path, status=sr.Status.AWAITING_INPUT)
        os.utime(path, ns=(self._FIXED_NS, self._FIXED_NS))

        second = scanner.scan()
        assert second[0].status == sr.Status.AWAITING_INPUT

    def test_removed_slug_leaves_no_stale_entry_for_a_different_record_at_same_slug(
        self, tmp_path
    ):
        """A slug removed from disk is absent from the next scan() result and
        leaves no stale cache entry behind: re-adding a DIFFERENT record at
        the exact same slug is picked up fresh, even in the adversarial case
        where its record.json is coincidentally stamped with the very same
        mtime the removed record's cache entry was keyed on -- proving the
        cache is keyed on the CURRENT dir set (evicted every scan), not just
        "mtime happened to differ this time."
        """
        import shutil

        from cockpit.registry_reader import SessionScanner

        first_record = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        sr.write_record(first_record, root=tmp_path)
        path = sr.record_path_for_slug('a-1', root=tmp_path)
        os.utime(path, ns=(self._FIXED_NS, self._FIXED_NS))

        scanner = SessionScanner(root=tmp_path)
        first = scanner.scan()
        assert {r.session_slug for r in first} == {'a-1'}

        shutil.rmtree(sr.sessions_dir(tmp_path) / 'a-1')
        assert scanner.scan() == []

        second_record = _make_record(session_slug='a-1', status=sr.Status.AWAITING_INPUT)
        sr.write_record(second_record, root=tmp_path)
        os.utime(path, ns=(self._FIXED_NS, self._FIXED_NS))  # coincidentally the same mtime

        second = scanner.scan()
        assert len(second) == 1
        assert second[0].status == sr.Status.AWAITING_INPUT

    def test_slug_unreadable_after_caching_is_skipped_not_stale_or_crashing(self, tmp_path):
        """Once a slug has been cached, its record.json can later become
        unreadable in two ways -- removed outright (the slug dir remains, so
        stat() itself raises) or overwritten with invalid JSON (read_record
        raises CorruptSessionRecord). Either way, scan() must fail-soft skip
        it: no crash from an unguarded stat(), and no stale cached record
        served in its place.
        """
        from cockpit.registry_reader import SessionScanner

        r1 = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        r2 = _make_record(session_slug='b-2', status=sr.Status.RUNNING)
        for r in (r1, r2):
            sr.write_record(r, root=tmp_path)

        scanner = SessionScanner(root=tmp_path)
        first = scanner.scan()
        assert {r.session_slug for r in first} == {'a-1', 'b-2'}

        # a-1: record.json vanishes but the slug directory remains.
        path_a = sr.record_path_for_slug('a-1', root=tmp_path)
        path_a.unlink()
        assert path_a.parent.is_dir()

        # b-2: record.json is overwritten with invalid JSON (a genuine
        # rewrite, so its mtime naturally advances).
        path_b = sr.record_path_for_slug('b-2', root=tmp_path)
        path_b.write_text('{not valid json')

        second = scanner.scan()
        assert second == []


class TestBuildSnapshot:
    def test_keyed_by_session_slug(self):
        from cockpit.registry_reader import build_snapshot

        records = [
            _make_record(session_slug='a-1', status=sr.Status.RUNNING),
            _make_record(session_slug='b-2', status=sr.Status.IDLE),
        ]

        snapshot = build_snapshot(records)

        assert set(snapshot.keys()) == {'a-1', 'b-2'}

    def test_excludes_start_ts_and_age(self):
        """Two records identical except start_ts must snapshot identically."""
        from cockpit.registry_reader import build_snapshot

        early = _make_record(session_slug='a-1', start_ts='2026-07-01T00:00:00+00:00')
        late = _make_record(session_slug='a-1', start_ts='2026-07-07T00:00:00+00:00')

        assert build_snapshot([early]) == build_snapshot([late])


class TestSnapshotChanged:
    def test_identical_record_sets_are_unchanged(self):
        from cockpit.registry_reader import build_snapshot, snapshot_changed

        records = [
            _make_record(session_slug='a-1', status=sr.Status.RUNNING),
            _make_record(session_slug='b-2', status=sr.Status.AWAITING_INPUT),
        ]

        old = build_snapshot(records)
        new = build_snapshot(records)

        assert snapshot_changed(old, new) is False

    def test_only_wall_clock_age_differing_is_unchanged(self):
        """Same records, only start_ts differs -> still no-change (age-independent)."""
        from cockpit.registry_reader import build_snapshot, snapshot_changed

        old = build_snapshot([_make_record(start_ts='2026-07-01T00:00:00+00:00')])
        new = build_snapshot([_make_record(start_ts='2026-07-07T00:00:00+00:00')])

        assert snapshot_changed(old, new) is False

    def test_added_record_is_changed(self):
        from cockpit.registry_reader import build_snapshot, snapshot_changed

        old = build_snapshot([_make_record(session_slug='a-1')])
        new = build_snapshot(
            [_make_record(session_slug='a-1'), _make_record(session_slug='b-2')]
        )

        assert snapshot_changed(old, new) is True

    def test_removed_record_is_changed(self):
        from cockpit.registry_reader import build_snapshot, snapshot_changed

        old = build_snapshot(
            [_make_record(session_slug='a-1'), _make_record(session_slug='b-2')]
        )
        new = build_snapshot([_make_record(session_slug='a-1')])

        assert snapshot_changed(old, new) is True

    def test_substantive_field_change_is_changed(self):
        """status running -> awaiting-input on the same slug must diff as changed."""
        from cockpit.registry_reader import build_snapshot, snapshot_changed

        old = build_snapshot([_make_record(session_slug='a-1', status=sr.Status.RUNNING)])
        new = build_snapshot(
            [_make_record(session_slug='a-1', status=sr.Status.AWAITING_INPUT)]
        )

        assert snapshot_changed(old, new) is True
