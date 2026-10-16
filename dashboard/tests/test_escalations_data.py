"""Tests for dashboard.data.escalations module.

Most tests use tmp_path to materialise fake escalation dirs (root + archive
subdirs) and synthesise minimal escalation-shaped dicts — no async / MCP
traffic.  The exception is :class:`TestFetchPinsRecovery` at the bottom, which
drives the ``fetch_pins_recovery`` escalation-URL fan-out over an
``httpx.MockTransport`` (same idiom as tests/test_merge_halt.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from shared.testing_virtual_clock import virtual_clock_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(esc_id: str, task_id: str = '1', level: int = 0, status: str = 'pending',
          worktree: str | None = None) -> dict:
    """Return a minimal escalation-shaped dict."""
    d = {'id': esc_id, 'task_id': task_id, 'level': level, 'status': status}
    if worktree is not None:
        d['worktree'] = worktree
    return d


def _named(roots: list[tuple[Path, list[dict]]]) -> list[tuple[str, Path, list[dict]]]:
    """Each ``(root, task_rows)`` as a ``resolve_owning_project`` candidate owned by its basename."""
    return [(root.name, root, rows) for root, rows in roots]


def _write_esc(directory: Path, filename: str, data: dict) -> Path:
    """Write escalation dict to a JSON file in directory."""
    path = directory / filename
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Tests for load_queue_escalations
# ---------------------------------------------------------------------------

class TestLoadQueueEscalations:
    """Tests for load_queue_escalations(esc_dir: Path) -> list[dict]."""

    def test_missing_directory_returns_empty(self, tmp_path):
        from dashboard.data.escalations import load_queue_escalations

        result = load_queue_escalations(tmp_path / 'nonexistent')
        assert result == []

    def test_non_directory_path_returns_empty(self, tmp_path):
        """A file (not a directory) at the given path returns []."""
        from dashboard.data.escalations import load_queue_escalations

        fake_dir = tmp_path / 'not_a_dir.json'
        fake_dir.write_text('{}')

        result = load_queue_escalations(fake_dir)
        assert result == []

    def test_empty_directory_returns_empty(self, tmp_path):
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        result = load_queue_escalations(esc_dir)
        assert result == []

    def test_root_level_json_files_are_loaded(self, tmp_path):
        """*.json files at the root of esc_dir are returned as dicts."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        esc1 = _esc('esc-1-1', task_id='1', level=0, status='pending')
        esc2 = _esc('esc-2-1', task_id='2', level=1, status='resolved')
        _write_esc(esc_dir, 'esc-1-1.json', esc1)
        _write_esc(esc_dir, 'esc-2-1.json', esc2)

        result = load_queue_escalations(esc_dir)
        assert len(result) == 2
        result_ids = {r['id'] for r in result}
        assert result_ids == {'esc-1-1', 'esc-2-1'}

    def test_archive_subdir_files_are_excluded(self, tmp_path):
        """Files in archive/YYYY-MM-DD/ subdirs are NOT included (root-only glob)."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        # Root-level file — should be included
        esc_root = _esc('esc-root-1', task_id='10')
        _write_esc(esc_dir, 'esc-root-1.json', esc_root)

        # Archive subtree — must NOT be included
        archive_day = esc_dir / 'archive' / '2026-05-27'
        archive_day.mkdir(parents=True)
        esc_archived = _esc('esc-archived-1', task_id='20')
        _write_esc(archive_day, 'esc-archived-1.json', esc_archived)

        result = load_queue_escalations(esc_dir)
        ids = {r['id'] for r in result}
        assert 'esc-root-1' in ids
        assert 'esc-archived-1' not in ids

    def test_malformed_json_is_skipped_and_warning_emitted(self, tmp_path, caplog):
        """A file with bad JSON is skipped; valid files are still returned.

        Verifies that logger.warning is called for the bad file.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        # Valid file
        esc_valid = _esc('esc-good-1', task_id='1')
        _write_esc(esc_dir, 'esc-good-1.json', esc_valid)

        # Malformed JSON
        (esc_dir / 'esc-bad-1.json').write_text('this is not json {{{')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.escalations'):
            result = load_queue_escalations(esc_dir)

        assert len(result) == 1
        assert result[0]['id'] == 'esc-good-1'
        assert caplog.records, 'Expected a WARNING log for the bad JSON file'
        assert any('esc-bad-1.json' in rec.message or 'esc-bad-1' in rec.message
                   for rec in caplog.records), 'WARNING should reference the bad file'

    def test_fields_intact(self, tmp_path):
        """Loaded dict has all original fields intact — nothing stripped or transformed."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()

        esc = _esc('esc-3-1', task_id='3', level=2, status='dismissed',
                   worktree='/home/leo/src/proj/.worktrees/3')
        esc['extra_field'] = 'should-survive'
        _write_esc(esc_dir, 'esc-3-1.json', esc)

        result = load_queue_escalations(esc_dir)
        assert len(result) == 1
        loaded = result[0]
        assert loaded['id'] == 'esc-3-1'
        assert loaded['task_id'] == '3'
        assert loaded['level'] == 2
        assert loaded['status'] == 'dismissed'
        assert loaded['worktree'] == '/home/leo/src/proj/.worktrees/3'
        assert loaded['extra_field'] == 'should-survive'

    def test_non_escalation_json_resident_is_not_read(self, tmp_path):
        """A well-formed non-``esc-*`` JSON file in the queue root is not an escalation.

        The queue directory has a second writer:
        ``orchestrator/src/orchestrator/b3_gate.py::STATE_REL_PATH`` keeps
        ``b3-state.json`` there.  Read as an escalation it becomes a row with
        no ``id``.  It is a permanent resident, not a read failure, so it is
        not reported in *skipped* either.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        good = _esc('esc-good-1', task_id='1', level=1, status='pending')
        _write_esc(esc_dir, 'esc-good-1.json', good)
        _write_esc(esc_dir, 'b3-state.json', {
            'launches': [],
            'charges': [{'task_id': '1615', 'charged_at': '2026-09-20T00:00:00+00:00'}],
        })

        skipped: list = []
        result = load_queue_escalations(esc_dir, skipped=skipped)

        assert result == [good]
        assert skipped == []

    # -- the opt-in ``skipped`` out-parameter -------------------------------
    #
    # Skipping an unparseable file is correct — one corrupt escalation must not
    # crash a queue scan.  Doing it with no channel back to the caller is not:
    # the reader holds both the path and the exception at the failure point and
    # threw both away into a log line no payload consumer can read (INV-2,
    # ``structured-facts-at-failure``).  ``skipped`` is that channel.

    def test_skipped_out_parameter_records_unparseable_files(self, tmp_path):
        """An opted-in caller learns WHICH file was dropped and WHY."""
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        _write_esc(esc_dir, 'esc-good-1.json', _esc('esc-good-1', task_id='1'))
        bad = esc_dir / 'esc-bad-1.json'
        bad.write_text('this is not json {{{')

        skipped: list = []
        result = load_queue_escalations(esc_dir, skipped=skipped)

        # The return value is untouched — this is a second channel, not a
        # change to what the reader yields.
        assert len(result) == 1
        assert result[0]['id'] == 'esc-good-1'
        assert len(skipped) == 1
        assert skipped[0]['path'] == bad
        assert isinstance(skipped[0]['error'], str) and skipped[0]['error']

    def test_skipped_records_os_errors_not_just_decode_errors(self, tmp_path):
        """BOTH arms of ``except (JSONDecodeError, OSError)`` report, not just JSON.

        ``Path.glob('esc-*.json')`` yields directories too, so a directory named
        ``esc-weird.json`` makes ``read_text()`` raise ``IsADirectoryError`` — a
        deterministic OSError needing no permission games or monkeypatching.
        A reader that only reported the decode arm would still lose every
        unreadable/permission-denied file silently, which is the more likely
        production failure of the two.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        _write_esc(esc_dir, 'esc-good-1.json', _esc('esc-good-1', task_id='1'))
        weird = esc_dir / 'esc-weird.json'
        weird.mkdir()

        skipped: list = []
        result = load_queue_escalations(esc_dir, skipped=skipped)

        assert [r['id'] for r in result] == ['esc-good-1']
        assert len(skipped) == 1
        assert skipped[0]['path'] == weird
        assert isinstance(skipped[0]['error'], str) and skipped[0]['error']

    def test_skipped_accumulator_is_appended_not_replaced(self, tmp_path):
        """Entries are APPENDED, so one list can span several queue dirs.

        ``build_escalation_queues`` calls this reader once per orchestrator
        root; a caller that wants the whole fleet's skips in one list must be
        able to reuse the accumulator rather than merge N of them.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        bad = esc_dir / 'esc-bad-1.json'
        bad.write_text('this is not json {{{')

        sentinel = {'path': tmp_path / 'from-an-earlier-dir.json', 'error': 'earlier'}
        skipped: list = [sentinel]
        load_queue_escalations(esc_dir, skipped=skipped)

        assert len(skipped) == 2
        assert skipped[0] is sentinel
        assert skipped[1]['path'] == bad

    def test_omitting_skipped_leaves_behaviour_unchanged(self, tmp_path, caplog):
        """The default path is byte-identical to today — the back-compat pin.

        Neither of the two ``build_escalation_queues`` call sites opts in, so
        for the escalation views the WARNING log stays the only signal.  (Named
        by function, not by line: a line-number citation in this file is stale
        the moment anything above it moves — the diff that added this test
        pushed those very calls down ~28 lines.)

        This asserts the un-opted-in call still returns only the valid record,
        still does not raise, and still logs — i.e. that the new keyword is
        additive and no existing caller had to change.
        """
        from dashboard.data.escalations import load_queue_escalations

        esc_dir = tmp_path / 'escalations'
        esc_dir.mkdir()
        _write_esc(esc_dir, 'esc-good-1.json', _esc('esc-good-1', task_id='1'))
        (esc_dir / 'esc-bad-1.json').write_text('this is not json {{{')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.escalations'):
            result = load_queue_escalations(esc_dir)

        assert len(result) == 1
        assert result[0]['id'] == 'esc-good-1'
        assert any('esc-bad-1' in rec.message for rec in caplog.records), \
            'the WARNING must survive for callers that do not opt in'


# ---------------------------------------------------------------------------
# Tests for resolve_owning_project — worktree-prefix arm (step 3)
# ---------------------------------------------------------------------------

class TestResolveOwningProjectWorktreeArm:
    """Tests for the worktree-prefix arm of resolve_owning_project."""

    def test_worktree_under_dot_worktrees_resolves_to_project(self, tmp_path):
        """Escalation with worktree under <root>/.worktrees/<id> resolves to root.name."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [])]

        esc = _esc('esc-1', task_id='42',
                   worktree=str(proj_a / '.worktrees' / '42'))
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'projA'

    def test_worktree_directly_under_root_resolves(self, tmp_path):
        """Worktree that starts with str(root) (not via .worktrees/) also resolves."""
        from dashboard.data.escalations import resolve_owning_project

        proj_b = tmp_path / 'projB'
        roots = [(proj_b, [])]

        esc = _esc('esc-2', task_id='55',
                   worktree=str(proj_b / 'some-subdir'))
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'projB'

    def test_dot_worktrees_prefix_form_also_matches(self, tmp_path):
        """Explicitly test the root/.worktrees/ prefix form resolves."""
        from dashboard.data.escalations import resolve_owning_project

        proj_c = tmp_path / 'projC'
        roots = [(proj_c, [])]

        # Worktree string exactly starts with str(proj_c / '.worktrees')
        wt = str(proj_c / '.worktrees' / '99')
        esc = _esc('esc-3', task_id='99', worktree=wt)
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'projC'

    def test_first_root_wins_when_multiple_could_match(self, tmp_path):
        """When multiple roots could prefix-match, the FIRST one in the list wins."""
        from dashboard.data.escalations import resolve_owning_project

        # proj_first contains proj_second as a subdirectory path-prefix-wise
        # (simulated by using parent/child paths)
        proj_first = tmp_path / 'workspace'
        proj_second = tmp_path / 'workspace' / 'sub'

        # worktree is under workspace/sub/.worktrees — both roots prefix-match
        # because proj_first path is a prefix of proj_second path
        wt = str(proj_second / '.worktrees' / '10')
        esc = _esc('esc-4', task_id='10', worktree=wt)

        roots = [(proj_first, []), (proj_second, [])]
        result = resolve_owning_project(esc, _named(roots))
        # first root wins
        assert result == 'workspace'

    def test_no_matching_root_returns_none(self, tmp_path):
        """Worktree that doesn't match any root prefix returns None."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [])]

        esc = _esc('esc-5', task_id='7',
                   worktree='/completely/different/path')
        result = resolve_owning_project(esc, _named(roots))
        assert result is None

    def test_missing_worktree_returns_none(self, tmp_path):
        """Escalation without worktree key returns None (no worktree arm match)."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [])]

        esc = _esc('esc-6', task_id='8')  # no worktree field
        result = resolve_owning_project(esc, _named(roots))
        assert result is None


# ---------------------------------------------------------------------------
# Tests for resolve_owning_project — task-map probe fallback (step 5)
# ---------------------------------------------------------------------------

class TestResolveOwningProjectTaskMapArm:
    """Tests for the task-map fallback arm of resolve_owning_project."""

    def test_task_id_in_second_roots_task_map_resolves(self, tmp_path):
        """Escalation with unmatched worktree but task_id in roots[1].task_map resolves."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        proj_b = tmp_path / 'projB'

        task_map_b = [{'id': 42, 'title': 'some task'}]
        roots = [(proj_a, []), (proj_b, task_map_b)]

        esc = _esc('esc-1', task_id='42', worktree='/no-match/path')
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'projB'

    def test_first_root_wins_when_multiple_task_maps_contain_same_task_id(self, tmp_path):
        """When multiple task maps have the same task_id, the first root wins."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        proj_b = tmp_path / 'projB'

        task_map_a = [{'id': 42, 'title': 'task in A'}]
        task_map_b = [{'id': 42, 'title': 'task in B'}]
        roots = [(proj_a, task_map_a), (proj_b, task_map_b)]

        esc = _esc('esc-1', task_id='42')
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'projA'

    def test_no_match_in_any_task_map_returns_none(self, tmp_path):
        """task_id not in any task map and no worktree match → None."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        roots = [(proj_a, [{'id': 99, 'title': 'other task'}])]

        esc = _esc('esc-1', task_id='55')
        result = resolve_owning_project(esc, _named(roots))
        assert result is None

    def test_task_id_string_vs_int_coercion(self, tmp_path):
        """task_id as string '42' matches task map entry with id=42 (int)."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        # task map has int id
        task_map_a = [{'id': 42, 'title': 'some task'}]
        roots = [(proj_a, task_map_a)]

        # esc.task_id is a string
        esc = _esc('esc-1', task_id='42')
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'projA'

    def test_the_owner_is_returned_as_passed_so_shared_basenames_stay_distinct(self, tmp_path):
        """Two roots named ``proj`` are two owners: the match is the candidate, never its name."""
        from dashboard.data.escalations import resolve_owning_project

        first = object()
        second = object()
        candidates = [
            (first, tmp_path / 'a' / 'proj', [{'id': 8}]),
            (second, tmp_path / 'b' / 'proj', [{'id': 7}]),
        ]

        assert resolve_owning_project(_esc('esc-1', task_id='7'), candidates) is second
        worktree = str(tmp_path / 'b' / 'proj' / '.worktrees' / '9')
        assert resolve_owning_project(
            _esc('esc-2', task_id='9', worktree=worktree), candidates,
        ) is second

    def test_no_worktree_falls_back_to_task_map(self, tmp_path):
        """Escalation without worktree field falls back to task map probe."""
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        task_map_a = [{'id': 7}]
        roots = [(proj_a, task_map_a)]

        esc = _esc('esc-1', task_id='7')  # no worktree key
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'projA'


# ---------------------------------------------------------------------------
# Tests for build_escalation_queues — subsections from the escalation corpus
# ---------------------------------------------------------------------------

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
"""The instant every corpus datum here is measured at."""


def _record(esc_id: str, task_id: str = '1', level: int = 0, status: str = 'pending',
            worktree: str | None = None) -> dict:
    """One escalation as the server writes it — parseable by the corpus walk."""
    from escalation.models import Escalation

    return Escalation(
        id=esc_id, task_id=task_id, agent_role='implementer', severity='blocking',
        category='design_concern', summary=f'summary of {esc_id}',
        timestamp='2026-09-01T00:00:00+00:00', status=status, level=level,
        worktree=worktree,
    ).to_dict()


def _put(directory: Path, record: dict, *, archived: bool = False) -> Path:
    target = directory / 'archive' / '2026-09-01' if archived else directory
    target.mkdir(parents=True, exist_ok=True)
    return _write_esc(target, f"{record['id']}.json", record)


def _config(primary: Path, extra: list[Path] | None = None):
    from dashboard.config import DashboardConfig

    primary.mkdir(parents=True, exist_ok=True)
    for root in extra or []:
        root.mkdir(parents=True, exist_ok=True)
    return DashboardConfig(project_root=primary, known_project_roots=extra or [])


def _queues_dir(root: Path) -> Path:
    return root / 'data' / 'escalations'


def _recon_dir(config) -> Path:
    return config.reconciliation_escalations_dir


def _build(config, active_rows: dict | None = None) -> dict:
    """``build_escalation_queues`` over a fresh walk of *config*'s corpus at NOW."""
    from dashboard.data.escalation_corpus import corpus_queues, measure_corpus
    from dashboard.data.escalations import build_escalation_queues

    corpus = measure_corpus(corpus_queues(config), now=NOW)
    return build_escalation_queues(corpus, active_rows=active_rows or {})


def _sub(result: dict, sub_id: str) -> dict:
    return next(s for s in result['subsections'] if s['id'] == sub_id)


class TestBuildEscalationQueuesSubsections:
    """Subsections built from the corpus: same shape, corpus_queues order, root rows only."""

    def test_one_subsection_per_queue_in_corpus_order(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [primary, reify])
        _put(_queues_dir(primary), _record('esc-1-1'))
        _put(_queues_dir(reify), _record('esc-2-1'))
        _put(_recon_dir(config), _record('esc-3-1'))

        result = _build(config)

        assert [(s['id'], s['label'], s['kind']) for s in result['subsections']] == [
            (str(primary.resolve()), 'primary', 'orchestrator'),
            (str(reify.resolve()), 'reify', 'orchestrator'),
            ('reconciliation', 'fused-memory', 'reconciliation'),
        ]
        for sub in result['subsections']:
            assert set(sub) >= {'id', 'label', 'kind', 'escalations', 'skipped', 'summary'}
            assert set(sub['summary']) == {'by_level', 'by_status', 'skipped_count'}

    def test_rows_are_the_escalation_fields_plus_project(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        record = _record('esc-1-1', task_id='7', level=1)
        _put(_queues_dir(primary), record)

        (row,) = _sub(_build(config), str(primary.resolve()))['escalations']

        assert row['project'] == 'primary'
        assert {k: v for k, v in row.items() if k not in ('project', 'project_root')} == record

    def test_archived_records_are_not_rows_but_count_in_open_in_history(self, tmp_path):
        from dashboard.data.escalation_corpus import EscalationView

        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_queues_dir(primary), _record('esc-1-1'))
        _put(_queues_dir(primary), _record('esc-1-2'), archived=True)

        sub = _sub(_build(config), str(primary.resolve()))

        assert [row['id'] for row in sub['escalations']] == ['esc-1-1']
        assert sub['views'][EscalationView.QUEUE_PENDING].value == 1
        assert sub['views'][EscalationView.OPEN_IN_HISTORY].value == 2

    def test_top_level_views_cover_every_queue_at_the_corpus_instant(self, tmp_path):
        from dashboard.data.datum import validate_datum
        from dashboard.data.escalation_corpus import EscalationView

        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        for n in (1, 2):
            _put(_queues_dir(primary), _record(f'esc-1-{n}'))
        for n in (3, 4, 5):
            _put(_queues_dir(primary), _record(f'esc-1-{n}'), archived=True)
        _put(_queues_dir(primary), _record('esc-1-6', status='resolved'))
        _put(_queues_dir(reify), _record('esc-2-1'))
        _put(_recon_dir(config), _record('esc-3-1'))

        views = _build(config)['views']

        assert views[EscalationView.QUEUE_PENDING].value == 4
        assert views[EscalationView.OPEN_IN_HISTORY].value == 7
        for datum in views.values():
            assert datum.as_of == NOW
            validate_datum(datum, NOW)

    def test_orchestrator_subsection_reports_skipped_files(self, tmp_path):
        """INV-2: a queue that reads fewer escalations than it holds says so in the payload."""
        primary = tmp_path / 'primary'
        config = _config(primary)
        esc_dir = _queues_dir(primary)
        _put(esc_dir, _record('esc-good-1'))
        (esc_dir / 'esc-bad-1.json').write_text('{not json')

        sub = _sub(_build(config), str(primary.resolve()))

        assert [e['id'] for e in sub['escalations']] == ['esc-good-1']
        (entry,) = sub['skipped']
        assert set(entry) == {'path', 'error', 'location'}
        assert isinstance(entry['path'], str) and entry['path'].endswith('esc-bad-1.json')
        assert isinstance(entry['error'], str) and entry['error']
        assert entry['location'] == 'root'
        assert sub['summary']['skipped_count'] == 1

    def test_an_unreadable_archive_file_is_skipped_with_its_location(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        archive = _queues_dir(primary) / 'archive' / '2026-09-01'
        archive.mkdir(parents=True)
        (archive / 'esc-bad-1.json').write_text('{not json')

        sub = _sub(_build(config), str(primary.resolve()))

        assert [e['location'] for e in sub['skipped']] == ['archive']
        assert sub['summary']['skipped_count'] == 1

    def test_skipped_is_per_subsection_not_shared(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        _put(_queues_dir(primary), _record('esc-1-1'))
        (_queues_dir(primary) / 'esc-bad-1.json').write_text('{not json')
        _put(_queues_dir(reify), _record('esc-2-1'))
        _put(_recon_dir(config), _record('esc-3-1'))

        result = _build(config)

        primary_id = str(primary.resolve())
        assert len(_sub(result, primary_id)['skipped']) == 1
        for sub in result['subsections']:
            if sub['id'] != primary_id:
                assert sub['skipped'] == []

    def test_reconciliation_subsection_reports_skipped_files(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_queues_dir(primary), _record('esc-1-1'))
        recon = _recon_dir(config)
        recon.mkdir(parents=True)
        (recon / 'esc-bad-1.json').write_text('{not json')
        (recon / 'esc-bad-2.json').write_text('also not json')

        result = _build(config)

        assert {Path(e['path']).name for e in _sub(result, 'reconciliation')['skipped']} == {
            'esc-bad-1.json', 'esc-bad-2.json',
        }
        assert _sub(result, str(primary.resolve()))['skipped'] == []

    def test_skipped_reports_os_errors_end_to_end(self, tmp_path):
        """The OSError arm reaches the payload too, not just a decode failure.

        A directory named like a queue file makes ``read_text()`` raise
        ``IsADirectoryError``. It is named ``esc-*.json`` because the corpus
        walks ``iter_all_escalation_paths``' ``esc-*.json`` glob — the
        documented population change from ``load_queue_escalations``' ``*.json``.
        """
        primary = tmp_path / 'primary'
        config = _config(primary)
        esc_dir = _queues_dir(primary)
        _put(esc_dir, _record('esc-1-1'))
        (esc_dir / 'esc-weird-1.json').mkdir()

        sub = _sub(_build(config), str(primary.resolve()))

        assert [e['id'] for e in sub['escalations']] == ['esc-1-1']
        (entry,) = sub['skipped']
        assert Path(entry['path']).name == 'esc-weird-1.json'
        assert entry['error']
        assert sub['summary']['skipped_count'] == 1

    def test_payload_with_skips_is_json_serializable(self, tmp_path):
        """What the API layer hands ``JSONResponse`` survives ``json.dumps``."""
        from dashboard.data.escalations import card_datums
        from dashboard.data.redux_api import shape_escalations

        primary = tmp_path / 'primary'
        config = _config(primary)
        esc_dir = _queues_dir(primary)
        _put(esc_dir, _record('esc-1-1'))
        (esc_dir / 'esc-bad-1.json').write_text('{not json')
        recon = _recon_dir(config)
        recon.mkdir(parents=True)
        (recon / 'esc-bad-2.json').write_text('also not json')

        queues = _build(config)
        shaped = shape_escalations(queues, card_datums(queues, {}), served_at=NOW)
        encoded = json.dumps(shaped)

        assert 'esc-bad-1.json' in encoded and 'esc-bad-2.json' in encoded
        assert shaped['ESCALATIONS']['summary']['skipped_count'] == 2


class TestOwnerAttribution:
    """Each row's owning root: its queue's, or — for reconciliation — worktree, then active rows."""

    def test_an_orchestrator_row_is_owned_by_its_queue_root(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        _put(_queues_dir(reify), _record('esc-2-1', task_id='9'))

        (row,) = _sub(_build(config), str(reify.resolve()))['escalations']

        assert row['project'] == 'reify'
        assert row['project_root'] == str(reify.resolve())

    def test_a_reconciliation_row_with_a_worktree_is_owned_by_that_root(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        worktree = str(reify / '.worktrees' / '9')
        _put(_recon_dir(config), _record('esc-9-1', task_id='9', worktree=worktree))

        (row,) = _sub(_build(config), 'reconciliation')['escalations']

        assert row['project'] == 'reify'
        assert row['project_root'] == str(reify.resolve())

    def test_a_reconciliation_row_without_worktree_is_owned_by_the_root_it_is_active_in(
        self, tmp_path,
    ):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        _put(_recon_dir(config), _record('esc-7-1', task_id='7'))
        active = {
            str(primary.resolve()): [{'id': 8}],
            str(reify.resolve()): [{'id': 7}],
        }

        (row,) = _sub(_build(config, active), 'reconciliation')['escalations']

        assert row['project'] == 'reify'
        assert row['project_root'] == str(reify.resolve())

    def test_the_first_root_wins_when_the_task_is_active_in_several(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        _put(_recon_dir(config), _record('esc-7-1', task_id='7'))
        active = {
            str(primary.resolve()): [{'id': 7}],
            str(reify.resolve()): [{'id': 7}],
        }

        (row,) = _sub(_build(config, active), 'reconciliation')['escalations']

        assert row['project_root'] == str(primary.resolve())

    def test_roots_sharing_a_basename_attribute_to_the_root_the_task_is_active_in(
        self, tmp_path,
    ):
        primary = tmp_path / 'a' / 'proj'
        other = tmp_path / 'b' / 'proj'
        config = _config(primary, [other])
        _put(_recon_dir(config), _record('esc-7-1', task_id='7'))
        active = {
            str(primary.resolve()): [{'id': 8}],
            str(other.resolve()): [{'id': 7}],
        }

        (row,) = _sub(_build(config, active), 'reconciliation')['escalations']

        assert row['project'] == 'proj'
        assert row['project_root'] == str(other.resolve())

    def test_a_reconciliation_row_active_nowhere_has_no_owner(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_recon_dir(config), _record('esc-7-1', task_id='7'))

        (row,) = _sub(_build(config, {str(primary.resolve()): [{'id': 8}]}),
                      'reconciliation')['escalations']

        assert row['project'] is None
        assert row['project_root'] is None


class TestCardTaskRefs:
    """The task ids a request must look up: every attributable row with a numeric id."""

    def test_refs_name_the_owning_root_and_the_numeric_id(self, tmp_path):
        from dashboard.data.escalations import card_task_refs
        from dashboard.data.task_lookup import TaskRef

        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        _put(_queues_dir(primary), _record('esc-1-1', task_id='1'))
        _put(_queues_dir(primary), _record('esc-x-1', task_id='not-a-number'))
        _put(_queues_dir(reify), _record('esc-2-1', task_id='2'))
        _put(_recon_dir(config), _record('esc-3-1', task_id='3'))
        _put(_recon_dir(config), _record('esc-4-1', task_id='4'))

        refs = card_task_refs(_build(config, {str(reify.resolve()): [{'id': 3}]}))

        assert refs == {
            TaskRef(str(primary.resolve()), 1),
            TaskRef(str(reify.resolve()), 2),
            TaskRef(str(reify.resolve()), 3),
        }


class TestCardDatums:
    """Every row gets a task Datum: the lookup's answer, or an unknown that says why."""

    def _fixture(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_queues_dir(primary), _record('esc-1-1', task_id='1'))
        _put(_queues_dir(primary), _record('esc-2-1', task_id='2'))
        _put(_queues_dir(primary), _record('esc-none-1', task_id=''))
        _put(_queues_dir(primary), _record('esc-x-1', task_id='abc'))
        _put(_recon_dir(config), _record('esc-7-1', task_id='7'))
        return primary, _build(config)

    def test_every_row_is_keyed_and_reasons_say_which_case(self, tmp_path):
        from dashboard.data.datum import Datum, DatumState
        from dashboard.data.escalations import card_datums
        from dashboard.data.task_lookup import TaskRef

        primary, queues = self._fixture(tmp_path)
        root = str(primary.resolve())
        found = Datum({'id': 1, 'title': 'one'}, NOW, DatumState.FRESH, None, 1200)

        cards = card_datums(queues, {TaskRef(root, 1): found})

        assert set(cards) == {
            (root, 'esc-1-1'), (root, 'esc-2-1'), (root, 'esc-none-1'),
            (root, 'esc-x-1'), ('reconciliation', 'esc-7-1'),
        }
        assert cards[(root, 'esc-1-1')] is found
        reasons = {key: cards[key].reason or '' for key in cards if key != (root, 'esc-1-1')}
        assert 'not looked up' in reasons[(root, 'esc-2-1')]
        assert 'no task id' in reasons[(root, 'esc-none-1')]
        assert 'task id is not a number' in reasons[(root, 'esc-x-1')]
        assert 'no owning project' in reasons[('reconciliation', 'esc-7-1')]
        for key, datum in cards.items():
            if key != (root, 'esc-1-1'):
                assert datum.state is DatumState.UNKNOWN
                assert datum.value is None

    def test_one_esc_id_in_two_queues_keeps_two_entries(self, tmp_path):
        from dashboard.data.escalations import card_datums

        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_queues_dir(primary), _record('esc-101-1', task_id='101'))
        _put(_recon_dir(config), _record('esc-101-1', task_id='101'))

        cards = card_datums(_build(config), {})

        assert set(cards) == {
            (str(primary.resolve()), 'esc-101-1'), ('reconciliation', 'esc-101-1'),
        }


# ---------------------------------------------------------------------------
# Tests for build_escalation_queues — summary counts
# ---------------------------------------------------------------------------

class TestBuildEscalationQueuesSummary:
    """Per-subsection and top-level summary bucketing over the live-queue rows."""

    def test_per_subsection_summary_shape(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_queues_dir(primary), _record('esc-1-1'))

        for sub in _build(config)['subsections']:
            assert set(sub['summary']['by_level']) == {0, 1, 2}
            assert set(sub['summary']['by_status']) == {'pending', 'resolved', 'dismissed'}

    def test_per_subsection_counts_correct(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        _put(_queues_dir(primary), _record('esc-1-1', level=0, status='pending'))
        _put(_queues_dir(primary), _record('esc-1-2', level=1, status='resolved'))
        _put(_queues_dir(primary), _record('esc-1-3', level=2, status='dismissed'))
        _put(_queues_dir(primary), _record('esc-1-4', level=2), archived=True)
        _put(_queues_dir(reify), _record('esc-2-1', level=1, status='pending'))

        result = _build(config)
        primary_sub = _sub(result, str(primary.resolve()))
        reify_sub = _sub(result, str(reify.resolve()))

        assert primary_sub['summary']['by_level'] == {0: 1, 1: 1, 2: 1}
        assert primary_sub['summary']['by_status'] == {
            'pending': 1, 'resolved': 1, 'dismissed': 1,
        }
        assert reify_sub['summary']['by_level'] == {0: 0, 1: 1, 2: 0}

    def test_top_level_summary_aggregates_all_subsections(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_queues_dir(primary), _record('esc-1-1', level=0, status='pending'))
        _put(_queues_dir(primary), _record('esc-1-2', level=1, status='resolved'))
        _put(_recon_dir(config), _record('esc-3-1', level=2, status='dismissed'))

        top = _build(config)['summary']

        assert top['by_level'] == {0: 1, 1: 1, 2: 1}
        assert top['by_status'] == {'pending': 1, 'resolved': 1, 'dismissed': 1}

    def test_unknown_level_and_status_excluded_from_buckets_but_in_list(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        _put(_queues_dir(primary), _record('esc-1-1', level=0, status='pending'))
        _put(_queues_dir(primary), _record('esc-1-2', level=3, status='weird'))

        sub = _sub(_build(config), str(primary.resolve()))

        assert {e['id'] for e in sub['escalations']} == {'esc-1-1', 'esc-1-2'}
        assert sum(sub['summary']['by_level'].values()) == 1
        assert sum(sub['summary']['by_status'].values()) == 1

    def test_skipped_count_sits_beside_the_counts_without_inflating_them(self, tmp_path):
        primary = tmp_path / 'primary'
        config = _config(primary)
        esc_dir = _queues_dir(primary)
        _put(esc_dir, _record('esc-1-1', level=1))
        (esc_dir / 'esc-bad-1.json').write_text('{not json')
        recon = _recon_dir(config)
        recon.mkdir(parents=True)
        (recon / 'esc-bad-2.json').write_text('also not json')

        result = _build(config)
        primary_sub = _sub(result, str(primary.resolve()))

        assert primary_sub['summary']['skipped_count'] == 1
        assert sum(primary_sub['summary']['by_level'].values()) == 1
        assert result['summary']['skipped_count'] == 2

    def test_skipped_count_zero_when_all_files_readable(self, tmp_path):
        primary = tmp_path / 'primary'
        reify = tmp_path / 'reify'
        config = _config(primary, [reify])
        _put(_queues_dir(primary), _record('esc-1-1'))

        result = _build(config)

        for sub in result['subsections']:
            assert sub['summary']['skipped_count'] == 0
        assert result['summary']['skipped_count'] == 0


# ---------------------------------------------------------------------------
# Regression tests for resolve_owning_project — worktree false-positive
# prefix matching (step 11)
# ---------------------------------------------------------------------------

class TestResolveOwningProjectPrefixRegression:
    """Regression tests locking the fix for str.startswith false-positive matches.

    The original implementation used raw ``str.startswith`` for path prefix
    matching.  This causes two classes of false positives:

    1. A root named ``workspace`` incorrectly matches a worktree under a
       *sibling* root ``workspace-2`` because the string ``".../workspace-2/..."``
       starts with ``".../workspace"``.
    2. A root's ``.worktrees`` prefix incorrectly matches paths under a
       ``.worktrees-archive`` sibling directory.

    All three tests below FAIL against the pre-fix ``str.startswith``
    implementation and PASS after the fix (``Path.is_relative_to``).
    """

    def test_sibling_root_prefix_does_not_false_match(self, tmp_path):
        """workspace-2 worktree must resolve to workspace-2, not workspace.

        Pre-fix bug: ``str(tmp_path/'workspace-2'/'.worktrees'/'42').startswith(
        str(tmp_path/'workspace'))`` is True because the string
        ".../workspace-2/..." starts with ".../workspace".
        """
        from dashboard.data.escalations import resolve_owning_project

        ws = tmp_path / 'workspace'
        ws2 = tmp_path / 'workspace-2'
        # workspace FIRST so the first-hit-wins rule amplifies the bug
        roots = [(ws, []), (ws2, [])]

        wt = str(ws2 / '.worktrees' / '42')
        esc = _esc('esc-reg-1', task_id='42', worktree=wt)
        result = resolve_owning_project(esc, _named(roots))
        assert result == 'workspace-2', (
            f"Expected 'workspace-2' but got {result!r} — "
            "sibling-prefix false positive not fixed"
        )

    def test_unrelated_sibling_worktree_returns_none(self, tmp_path):
        """Worktree under workspace-extra must NOT match root workspace.

        Pre-fix bug: ``".../workspace-extra/...".startswith(".../workspace")``
        is True.
        """
        from dashboard.data.escalations import resolve_owning_project

        ws = tmp_path / 'workspace'
        ws_extra = tmp_path / 'workspace-extra'
        roots = [(ws, [])]

        wt = str(ws_extra / '.worktrees' / '42')
        esc = _esc('esc-reg-2', task_id='42', worktree=wt)
        result = resolve_owning_project(esc, _named(roots))
        assert result is None, (
            f"Expected None but got {result!r} — "
            "sibling-prefix false positive not fixed"
        )

    def test_dot_worktrees_archive_suffix_does_not_false_match(self, tmp_path):
        """A sibling dir named .worktrees-archive must NOT match a root named .worktrees.

        The root here is ``projA/.worktrees`` (the worktrees directory itself).
        Its sibling ``projA/.worktrees-archive`` shares the string prefix
        ``.../projA/.worktrees`` but is NOT a child of root — it is a sibling.

        Pre-fix bug: ``str(projA/'.worktrees-archive'/'42').startswith(
        str(projA/'.worktrees'))`` is True because the raw string
        ``.worktrees-archive`` starts with ``.worktrees``.

        Post-fix: ``Path(...).is_relative_to(Path(projA/'.worktrees'))`` is False
        because ``.worktrees-archive`` is a different path component from ``.worktrees``.
        """
        from dashboard.data.escalations import resolve_owning_project

        proj_a = tmp_path / 'projA'
        # The ROOT is the .worktrees dir itself — the path to resolve against.
        worktrees_root = proj_a / '.worktrees'
        roots = [(worktrees_root, [])]

        # Sibling of worktrees_root: .worktrees-archive (same level, different name).
        # Its name string-starts-with ".worktrees" but it is NOT under worktrees_root.
        wt = str(proj_a / '.worktrees-archive' / '42')
        esc = _esc('esc-reg-3', task_id='42', worktree=wt)
        result = resolve_owning_project(esc, _named(roots))
        assert result is None, (
            f"Expected None but got {result!r} — "
            ".worktrees-archive false-matched .worktrees root via string prefix"
        )


# ---------------------------------------------------------------------------
# fetch_pins_recovery — escalation-URL fan-out (task 3543 step-23, spec S8)
# ---------------------------------------------------------------------------
#
# `get_pending_escalations` now computes a per-record `pins_recovery` list
# (escalation/src/escalation/server.py) and deliberately OMITS the key when it
# could not be computed.  These tests pin the dashboard-side fan-out that reads
# it, and in particular its THREE-state discipline, which mirrors
# `escalation.pins.PinReport.store_unavailable`:
#
#   None  — this project could not be read (transport error, timeout, error
#           envelope).  UNKNOWN.  The UI must render nothing.
#   {}    — read succeeded; no record carried an annotation.
#   {id: [task_ids]} — read succeeded and these records are annotated.
#
# Collapsing the first into the second is the exact esc-3163 defect: an empty
# map reads as "nothing pins this", which routes a genuinely-pinned strand down
# the wrong branch.  The same discipline applies per RECORD: an id absent from
# a non-None map is unknown, never "does not pin".


def _init_response(request_id: int = 1) -> httpx.Response:
    """Minimal MCP `initialize` reply so McpSession completes its handshake."""
    return httpx.Response(
        200,
        json={
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {
                'protocolVersion': '2025-03-26',
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'test', 'version': '0.1'},
            },
        },
        headers={'mcp-session-id': 'test-session-id'},
    )


def _tool_response(inner, request_id: int = 1) -> httpx.Response:
    """Encode *inner* the way FastMCP encodes a list-returning tool's result.

    Verified against the installed fastmcp 3.2.2
    (``fastmcp/tools/base.py::_convert_to_content``): a non-empty ``list`` of
    plain dicts is aggregated into a SINGLE TextContent block holding the
    JSON-serialised list, while an EMPTY list short-circuits to an empty
    ``content`` array (the ``all(isinstance(...))`` guard is vacuously true for
    ``[]``, so the list itself is returned as the content blocks).

    That asymmetry is load-bearing for this fan-out: the dashboard's
    ``_extract_tool_result`` maps empty content to ``{}``, so an authoritative
    "no pending escalations" arrives as a dict, not as a list.  A fixture that
    always emitted a text block would test a wire shape the real server never
    produces and would leave that branch unexercised.
    """
    if isinstance(inner, list) and not inner:
        content = []
    else:
        content = [{'type': 'text', 'text': json.dumps(inner)}]
    return httpx.Response(
        200,
        json={'jsonrpc': '2.0', 'id': request_id, 'result': {'content': content}},
        headers={'mcp-session-id': 'test-session-id'},
    )


class _PinsHandler:
    """MockTransport handler dispatching per-port `get_pending_escalations`.

    ``responses`` maps port → the value that port's escalation server returns
    (normally a ``list[dict]``).  ``fail_ports`` raise ``httpx.ConnectError``;
    ``slow_ports`` sleep first so the per-call timeout path can be driven.
    Every tools/call is recorded in ``tool_calls`` as ``(port, name, args)``.
    """

    def __init__(
        self,
        responses: dict[int, object] | None = None,
        *,
        fail_ports: set[int] | None = None,
        slow_ports: dict[int, float] | None = None,
    ):
        self.responses: dict[int, object] = responses or {}
        self.fail_ports = fail_ports or set()
        self.slow_ports = slow_ports or {}
        self.tool_calls: list[tuple[int, str, dict]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port
        assert port is not None
        if port in self.fail_ports:
            raise httpx.ConnectError('refused')
        if port in self.slow_ports:
            await asyncio.sleep(self.slow_ports[port])
        body = json.loads(request.content)
        method = body.get('method', '')
        request_id = body.get('id', 1)
        if method == 'initialize':
            return _init_response(request_id)
        if method.startswith('notifications/'):
            return httpx.Response(202, headers={'mcp-session-id': 'test-session-id'})
        params = body.get('params') or {}
        self.tool_calls.append(
            (port, params.get('name', ''), params.get('arguments') or {}),
        )
        return _tool_response(self.responses.get(port, []), request_id)


class _ExplodingHandler:
    """Handler that fails the test if any HTTP request is issued at all."""

    def __init__(self):
        self.called = False

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.called = True
        raise AssertionError(f'unexpected HTTP request to {request.url}')


def _pins_urls(*ports: int) -> dict[str, str]:
    return {f'proj{p}': f'http://127.0.0.1:{p}/mcp' for p in ports}


def _rec(esc_id: str, **extra) -> dict:
    """A compact pending-escalation record as the escalation server returns it."""
    d = {'id': esc_id, 'task_id': '3543', 'level': 1, 'status': 'pending'}
    d.update(extra)
    return d


class TestFetchPinsRecovery:
    """`fetch_pins_recovery(client, escalation_urls)` fan-out contract."""

    @pytest.fixture(autouse=True)
    def _clean_sessions(self):
        from dashboard.data.memory import reset_sessions
        reset_sessions()
        yield
        reset_sessions()

    async def test_empty_urls_short_circuits_without_any_http_call(self):
        """No configured escalation URLs → `{}` and not a single request."""
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _ExplodingHandler()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, {})
        assert result == {}
        assert handler.called is False

    async def test_annotated_records_map_id_to_task_ids(self):
        """A successful read returns `{esc_id: pins_recovery}` per project."""
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler({
            8100: [
                _rec('esc-a', pins_recovery=['3543']),
                _rec('esc-b', pins_recovery=[]),
            ],
            8105: [_rec('esc-c', task_id='99', pins_recovery=['99'])],
        })
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8100, 8105))

        assert set(result) == {'proj8100', 'proj8105'}
        assert result['proj8100'] == {'esc-a': ['3543'], 'esc-b': []}
        assert result['proj8105'] == {'esc-c': ['99']}

    async def test_requests_compact_pending_records(self):
        """The fan-out asks for `get_pending_escalations(compact=True)`.

        `_COMPACT_PENDING_FIELDS` (escalation/server.py) exists precisely
        because the dashboard reads compact records — it re-adds
        `pins_recovery` on top of the shared compact projection.  Requesting
        the full record would haul detail/members/options across the wire on
        every poll for a field this caller never reads.
        """
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler({8100: [_rec('esc-a', pins_recovery=['3543'])]})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await fetch_pins_recovery(client, _pins_urls(8100))

        assert len(handler.tool_calls) == 1, handler.tool_calls
        _port, name, args = handler.tool_calls[0]
        assert name == 'get_pending_escalations'
        assert args.get('compact') is True

    async def test_per_call_budget_reaches_the_request_timeout_too(self):
        """The probe's budget is threaded into `mcp_tool_call`, not only
        into the surrounding `asyncio.wait_for`.

        The two layers are complementary, not redundant — `mcp_tool_call`'s
        docstring (dashboard/data/memory.py) is explicit: the per-request
        value is handed to `client.post`, so it bounds connect/read/write
        **and pool acquisition**.  That last one is why threading it matters.
        With only the outer `wait_for`, a probe working to a 2.0s budget still
        leaves the inner call on httpx's 10s default while it waits for a free
        connection slot, so a saturated pool blows the poll cycle's budget by
        5x before the outer timeout can even be observed by the transport.

        The sibling this probe is modelled on, `task_runtime._probe_one`
        (dashboard/data/task_runtime.py:53-58), passes BOTH; this is the seam
        where the two must read identically.
        """
        from unittest.mock import AsyncMock, patch

        from dashboard.data.escalations import fetch_pins_recovery

        handler = _ExplodingHandler()  # every call is intercepted below
        transport = httpx.MockTransport(handler)
        stub = AsyncMock(return_value=[])
        async with httpx.AsyncClient(transport=transport) as client:
            with patch('dashboard.data.escalations.mcp_tool_call', stub):
                await fetch_pins_recovery(
                    client, _pins_urls(8100), per_call_timeout=2.0,
                )

        assert stub.await_count == 1, stub.await_args_list
        call = stub.await_args
        assert call is not None, 'mcp_tool_call was never awaited'
        # Assert on the kwarg SPECIFICALLY.  Pinning the whole call signature
        # would churn on any unrelated argument change without adding signal.
        assert call.kwargs.get('timeout') == 2.0, (
            'the per-call budget must reach mcp_tool_call so it bounds pool '
            f'acquisition too; got kwargs={call.kwargs!r}'
        )
        # ...and the existing request shape is carried through unchanged.
        assert call.args[2] == 'get_pending_escalations'
        assert call.args[3] == {'compact': True}

    async def test_record_without_the_key_is_omitted_not_defaulted(self):
        """An older escalation server omits `pins_recovery` → omit the id.

        Defaulting the missing key to `[]` would manufacture a confident
        "this record pins nothing" out of a server that never computed the
        annotation — the same false-negative the escalation side refuses to
        emit.  The id must simply be absent from the map so downstream renders
        UNKNOWN.
        """
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler({
            8100: [
                _rec('esc-old'),                            # pre-3543 server
                _rec('esc-new', pins_recovery=['3543']),
            ],
        })
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8100))

        annotated = result['proj8100']
        assert annotated is not None  # a successful read, not the UNKNOWN state
        assert annotated == {'esc-new': ['3543']}
        assert 'esc-old' not in annotated

    async def test_non_list_annotation_is_omitted(self):
        """A malformed `pins_recovery` (not a list) is dropped, not coerced."""
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler({
            8100: [
                _rec('esc-bad', pins_recovery='3543'),
                _rec('esc-null', pins_recovery=None),
                _rec('esc-ok', pins_recovery=['3543']),
            ],
        })
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8100))

        assert result['proj8100'] == {'esc-ok': ['3543']}

    async def test_authoritative_empty_queue_maps_to_empty_dict(self):
        """Zero pending escalations is a SUCCESSFUL read → `{}`, never None.

        FastMCP encodes an empty list as empty `content`, which the shared
        `_extract_tool_result` collapses to `{}` — so this arrives as a dict
        even though the tool returns a list.  It must still be distinguishable
        from an unreachable project.
        """
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler({8100: []})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8100))

        assert result['proj8100'] is not None
        assert result['proj8100'] == {}

    async def test_connect_error_maps_to_none_not_empty(self):
        """An unreachable project is UNKNOWN (None), not "nothing pins" (`{}`)."""
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler(fail_ports={8102})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8102))

        assert 'proj8102' in result
        assert result['proj8102'] is None

    @virtual_clock_test
    async def test_timeout_maps_to_none(self):
        """A project that does not answer inside per_call_timeout is UNKNOWN,
        while a sibling that did answer keeps its read.

        The answering project must beat the same deadline the slow one misses;
        on the host clock a stalled worker made it miss that deadline too, so
        this is a shared.testing_virtual_clock.virtual_clock_test.
        """
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler(
            {8100: [_rec('esc-a', pins_recovery=['3543'])], 8105: []},
            slow_ports={8105: 0.5},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(
                client, _pins_urls(8100, 8105), per_call_timeout=0.05,
            )

        assert result['proj8100'] == {'esc-a': ['3543']}
        assert result['proj8105'] is None

    async def test_error_envelope_maps_to_none(self):
        """A tool result that is a dict (error envelope) is UNKNOWN, not empty.

        The tool's contract is `list[dict]`; anything else means the call did
        not deliver the annotation.  The one exception — an EMPTY dict, which
        is how FastMCP+`_extract_tool_result` render an authoritative empty
        list — is covered by its own test above.
        """
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler({8100: {'error': 'no escalation queue wired'}})
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8100))

        assert result['proj8100'] is None

    async def test_one_project_failure_never_sinks_the_others(self):
        """Per-project isolation: every configured label is always present."""
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler(
            {
                8100: [_rec('esc-a', pins_recovery=['3543'])],
                8107: [],
            },
            fail_ports={8102},
        )
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8100, 8102, 8107))

        assert set(result) == {'proj8100', 'proj8102', 'proj8107'}
        assert result['proj8100'] == {'esc-a': ['3543']}
        assert result['proj8102'] is None
        assert result['proj8107'] == {}

    async def test_unanticipated_exception_sinks_only_its_own_project(self):
        """An exception class the probe does NOT catch degrades one project.

        ``_fetch_pins_one`` catches the transport family it can anticipate
        ((TimeoutError, httpx.HTTPError, OSError, ValueError)), but
        ``mcp_tool_call`` reaches ``McpSession.call_tool``, whose failure modes
        are not contractually narrowed to those.  If such an escape propagated
        out of the gather, app.py's outer ``except Exception`` would blank the
        annotation for EVERY project at once — the fleet-wide collapse this
        per-project fan-out exists to prevent.  A RuntimeError stands in for
        the whole unanticipated class.
        """
        from unittest.mock import patch

        from dashboard.data.escalations import fetch_pins_recovery

        async def _mcp(_client, base_url, _tool, _args, **_kwargs):
            # **_kwargs absorbs the per-request `timeout=` the probe threads
            # through; this test is about exception isolation, not the call
            # shape (which test_per_call_budget_reaches_the_request_timeout_too
            # pins).
            if '8102' in base_url:
                raise RuntimeError('session state went sideways')
            return [_rec('esc-a', pins_recovery=['3543'])]

        handler = _ExplodingHandler()  # every call is intercepted below
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with patch(
                'dashboard.data.escalations.mcp_tool_call', side_effect=_mcp,
            ):
                result = await fetch_pins_recovery(
                    client, _pins_urls(8100, 8102, 8107),
                )

        assert set(result) == {'proj8100', 'proj8102', 'proj8107'}, (
            'every configured label must still be present; got '
            f'{sorted(result)!r}'
        )
        assert result['proj8102'] is None, (
            'the raising project degrades to UNKNOWN, not to an empty map'
        )
        assert result['proj8100'] == {'esc-a': ['3543']}, (
            "a sibling's unanticipated exception must not blank this "
            f"project's annotation; got {result['proj8100']!r}"
        )
        assert result['proj8107'] == {'esc-a': ['3543']}

    async def test_records_that_are_not_dicts_do_not_raise(self):
        """A ragged list (strings/None mixed in) degrades instead of raising.

        This runs inside a dashboard poll cycle; a TypeError here would 500 the
        endpoint on account of one malformed record.
        """
        from dashboard.data.escalations import fetch_pins_recovery

        handler = _PinsHandler({
            8100: ['not-a-record', None, _rec('esc-ok', pins_recovery=['3543']), {}],
        })
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_pins_recovery(client, _pins_urls(8100))

        assert result['proj8100'] == {'esc-ok': ['3543']}
