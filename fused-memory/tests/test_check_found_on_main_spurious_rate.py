"""Tests for check_found_on_main_spurious_rate.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_audit_found_on_main_provenance.py / test_correct_found_on_main_backlog.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'check_found_on_main_spurious_rate.py'
)


def _load_module() -> types.ModuleType:
    mod_name = 'check_found_on_main_spurious_rate'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
parse_since = _mod.parse_since
find_spurious_since = _mod.find_spurious_since
format_summary = _mod.format_summary


# ---------------------------------------------------------------------------
# Fixture builders — shaped like build_audit_report's return value /
# get_tasks()'s raw task dicts.
# ---------------------------------------------------------------------------

def _detail(task_id: str, verdict: str, *, commit: str = 'a' * 40) -> dict:
    return {'task_id': task_id, 'verdict': verdict, 'commit': commit, 'reasons': []}


def _report(tasks: list[dict], ref: str = 'main') -> dict:
    return {'ref': ref, 'dry_run': True, 'total': len(tasks), 'tasks': tasks}


def _task(task_id: str, updated_at: str) -> dict:
    return {'id': task_id, 'updatedAt': updated_at}


SINCE = datetime(2026, 7, 16, 0, 0, 0, tzinfo=UTC)
BEFORE_SINCE = '2026-07-15T23:00:00Z'
AFTER_SINCE = '2026-07-16T01:00:00Z'


# ===========================================================================
# parse_since — pure ISO-8601 parsing
# ===========================================================================

class TestParseSince:
    def test_z_suffix_parsed_as_utc(self):
        dt = parse_since('2026-07-16T00:00:00Z')
        assert dt == SINCE
        assert dt.tzinfo is not None

    def test_offset_form_converted_to_utc(self):
        dt = parse_since('2026-07-16T02:00:00+02:00')
        assert dt == SINCE

    def test_naive_value_treated_as_utc(self):
        dt = parse_since('2026-07-16T00:00:00')
        assert dt == SINCE

    def test_unparseable_value_raises(self):
        with pytest.raises(ValueError):
            parse_since('not-a-date')


# ===========================================================================
# find_spurious_since — the core filter
# ===========================================================================

class TestFindSpuriousSince:
    def test_no_findings_when_report_is_clean(self):
        report = _report([_detail('100', 'ok')])
        tasks = [_task('100', AFTER_SINCE)]
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_flagged_verdict_before_since_is_excluded(self):
        report = _report([_detail('101', 'misattributed')])
        tasks = [_task('101', BEFORE_SINCE)]
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_flagged_verdict_after_since_is_included(self):
        report = _report([_detail('102', 'misattributed', commit='b' * 40)])
        tasks = [_task('102', AFTER_SINCE)]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert len(offenders) == 1
        assert offenders[0]['task_id'] == '102'
        assert offenders[0]['commit'] == 'b' * 40
        assert offenders[0]['verdict'] == 'misattributed'

    def test_deliverable_absent_after_since_is_included(self):
        report = _report([_detail('103', 'deliverable_absent')])
        tasks = [_task('103', AFTER_SINCE)]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert len(offenders) == 1
        assert offenders[0]['verdict'] == 'deliverable_absent'

    def test_narrower_than_flagged_verdicts_reverted_and_commit_not_on_main_excluded(self):
        # These are real audit verdicts, and real _FLAGGED_VERDICTS members,
        # but out of this predicate's narrower contract (see module
        # docstring) — must never appear in offenders even when fresh.
        report = _report([
            _detail('104', 'reverted'),
            _detail('105', 'commit_not_on_main'),
        ])
        tasks = [_task('104', AFTER_SINCE), _task('105', AFTER_SINCE)]
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_missing_updated_at_is_conservatively_excluded(self):
        report = _report([_detail('106', 'misattributed')])
        tasks = [{'id': '106'}]  # no updatedAt key at all
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_unparseable_updated_at_is_conservatively_excluded(self):
        report = _report([_detail('107', 'misattributed')])
        tasks = [_task('107', 'not-a-timestamp')]
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_task_missing_from_tasks_list_is_excluded(self):
        report = _report([_detail('108', 'misattributed')])
        assert find_spurious_since(report, [], SINCE) == []

    def test_results_sorted_by_numeric_task_id(self):
        report = _report([
            _detail('200', 'misattributed'),
            _detail('50', 'deliverable_absent'),
        ])
        tasks = [_task('200', AFTER_SINCE), _task('50', AFTER_SINCE)]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert [o['task_id'] for o in offenders] == ['50', '200']

    def test_multiple_found_on_main_tasks_only_fresh_flagged_ones_surface(self):
        report = _report([
            _detail('1', 'ok'),
            _detail('2', 'misattributed'),  # before since -> excluded
            _detail('3', 'misattributed'),  # after since -> included
            _detail('4', 'unverifiable'),   # not a spurious verdict
        ])
        tasks = [
            _task('1', AFTER_SINCE),
            _task('2', BEFORE_SINCE),
            _task('3', AFTER_SINCE),
            _task('4', AFTER_SINCE),
        ]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert [o['task_id'] for o in offenders] == ['3']


# ===========================================================================
# format_summary — one line per offender
# ===========================================================================

class TestFormatSummary:
    def test_empty_offenders_yields_no_lines(self):
        assert format_summary([]) == []

    def test_one_line_per_offender_carries_task_id_commit_and_flag_class(self):
        offenders = [
            {'task_id': '3', 'commit': 'c' * 40, 'verdict': 'misattributed'},
        ]
        lines = format_summary(offenders)
        assert len(lines) == 1
        assert 'task_id=3' in lines[0]
        assert f'commit={"c" * 40}' in lines[0]
        assert 'flag_class=misattributed' in lines[0]


# ===========================================================================
# _run() end-to-end CLI wiring — exit-code contract (both exit paths)
# ===========================================================================

class _FakeGitFacts:
    def __init__(self, project_root):
        self.project_root = project_root


def _install_fake_audit_module(monkeypatch, report):
    async def _fake_build_audit_report(tasks, git, ref='main'):
        return report

    fake_mod = types.ModuleType('audit_found_on_main_provenance')
    fake_mod.build_audit_report = _fake_build_audit_report  # type: ignore[attr-defined]
    fake_mod.GitFacts = _FakeGitFacts  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'audit_found_on_main_provenance', fake_mod)


class _FakeTaskmasterConfig:
    """Truthy sentinel — _run() only checks `config.taskmaster is None`."""


class _FakeFusedMemoryConfigWithTaskmaster:
    def __init__(self, *args, **kwargs):
        self.taskmaster = _FakeTaskmasterConfig()


class _FakeFusedMemoryConfigWithoutTaskmaster:
    def __init__(self, *args, **kwargs):
        self.taskmaster = None


class _FakeRunBackend:
    """Read-only fake: only start/close/get_tasks are exercised by _run();
    this script never calls update_task/set_task_status, so those surfaces
    aren't even stubbed here — their absence is itself a read-only guard."""

    def __init__(self, taskmaster_config=None, *, tasks=None):
        self.taskmaster_config = taskmaster_config
        self.started = False
        self.closed = False
        self.get_tasks_calls: list[str] = []
        self._tasks = tasks if tasks is not None else []

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def get_tasks(self, project_root):
        self.get_tasks_calls.append(project_root)
        return {'tasks': self._tasks}


def _install_fake_backend(monkeypatch, tasks):
    backend_holder: dict[str, _FakeRunBackend] = {}

    def _make_backend(taskmaster_config):
        backend_holder['backend'] = _FakeRunBackend(taskmaster_config, tasks=tasks)
        return backend_holder['backend']

    monkeypatch.setattr(
        'fused_memory.backends.sqlite_task_backend.SqliteTaskBackend', _make_backend,
    )
    return backend_holder


@pytest.mark.asyncio
class TestRunCliWiring:
    """_run() end-to-end: config load, the sibling audit import, backend
    start/close, report build, and both exit-code paths of the predicate
    contract — exit 0 (clean) and exit 1 (offenders present)."""

    async def test_exit_zero_when_no_fresh_flagged_tasks(self, monkeypatch):
        report = _report([_detail('9999', 'misattributed')])
        tasks = [_task('9999', BEFORE_SINCE)]  # stamp predates --since
        _install_fake_audit_module(monkeypatch, report)
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithTaskmaster,
        )
        backend_holder = _install_fake_backend(monkeypatch, tasks)

        args = argparse.Namespace(
            project_root='/proj', config=None, ref='main', since='2026-07-16T00:00:00Z',
        )
        exit_code = await _mod._run(args)

        assert exit_code == 0
        backend = backend_holder['backend']
        assert backend.started is True
        assert backend.closed is True
        assert backend.get_tasks_calls == ['/proj']

    async def test_exit_one_when_post_since_misattributed_stamp_present(self, monkeypatch):
        report = _report([_detail('8888', 'misattributed', commit='d' * 40)])
        tasks = [_task('8888', AFTER_SINCE)]  # stamp is fresh
        _install_fake_audit_module(monkeypatch, report)
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithTaskmaster,
        )
        backend_holder = _install_fake_backend(monkeypatch, tasks)

        args = argparse.Namespace(
            project_root='/proj', config=None, ref='main', since='2026-07-16T00:00:00Z',
        )
        exit_code = await _mod._run(args)

        assert exit_code == 1
        backend = backend_holder['backend']
        assert backend.started is True
        assert backend.closed is True  # closed even on the non-zero exit path

    async def test_missing_taskmaster_config_returns_1_without_creating_backend(
        self, monkeypatch,
    ):
        report = _report([])
        _install_fake_audit_module(monkeypatch, report)
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithoutTaskmaster,
        )
        created: list[int] = []
        monkeypatch.setattr(
            'fused_memory.backends.sqlite_task_backend.SqliteTaskBackend',
            lambda *a, **kw: created.append(1),  # noqa: ARG005
        )

        args = argparse.Namespace(
            project_root='/proj', config=None, ref='main', since='2026-07-16T00:00:00Z',
        )
        exit_code = await _mod._run(args)

        assert exit_code == 1
        assert created == []

    async def test_malformed_since_raises_valueerror(self, monkeypatch):
        # parse_since(args.since) runs before any backend/report work, so a
        # bad --since surfaces as a ValueError straight out of _run() — it
        # is main()'s job (tested below) to catch this and map it to the
        # distinct usage-error exit code rather than 0/1.
        _install_fake_audit_module(monkeypatch, _report([]))
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithTaskmaster,
        )

        args = argparse.Namespace(
            project_root='/proj', config=None, ref='main', since='not-a-date',
        )

        with pytest.raises(ValueError):
            await _mod._run(args)


# ===========================================================================
# main() — malformed --since maps to a distinct usage-error exit code (2),
# never the business-logic 0/1 (see module docstring "Contract").
# ===========================================================================

class TestMainMalformedSinceExitCode:
    def test_malformed_since_exits_with_distinct_usage_code(self, monkeypatch, capsys):
        _install_fake_audit_module(monkeypatch, _report([]))
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithTaskmaster,
        )
        monkeypatch.setattr(
            sys, 'argv',
            [
                'check_found_on_main_spurious_rate.py',
                '--since', 'not-a-date',
                '--project-root', '/proj',
            ],
        )

        exit_code = _mod.main()

        assert exit_code == 2
        captured = capsys.readouterr()
        assert 'not-a-date' in captured.err


# ===========================================================================
# Regression guard for the deferred sibling import inside _run()
# ===========================================================================

class TestSiblingAuditImportResolves:
    def test_sibling_module_exposes_build_audit_report_and_gitfacts(self, monkeypatch):
        monkeypatch.delitem(sys.modules, 'audit_found_on_main_provenance', raising=False)
        monkeypatch.syspath_prepend(str(SCRIPT_PATH.parent))

        import importlib  # noqa: PLC0415
        sibling = importlib.import_module('audit_found_on_main_provenance')

        assert callable(sibling.build_audit_report)
        assert callable(sibling.GitFacts)
