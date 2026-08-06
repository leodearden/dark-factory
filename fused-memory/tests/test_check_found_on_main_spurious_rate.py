"""Tests for check_found_on_main_spurious_rate.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_audit_found_on_main_provenance.py / test_correct_found_on_main_backlog.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


def _stamped_task(
    task_id: str,
    updated_at: str,
    stamped_at: str | None,
    *,
    commit: str = 'a' * 40,
    as_json: bool = False,
) -> dict:
    """A raw task dict carrying a found_on_main done_provenance blob.

    ``stamped_at=None`` models a legacy (pre-3576) stamp: the key is absent
    from the blob entirely, which is the load-bearing "predates the field"
    signal. ``as_json`` renders metadata as a JSON string rather than a
    dict — both shapes reach this code in practice and parse_metadata
    accepts either.
    """
    provenance: dict = {
        'kind': 'found_on_main',
        'commit': commit,
        'note': f'sibling landed this for {task_id}',
    }
    if stamped_at is not None:
        provenance['stamped_at'] = stamped_at
    metadata: dict = {'done_provenance': provenance}
    return {
        'id': task_id,
        'updatedAt': updated_at,
        'metadata': json.dumps(metadata) if as_json else metadata,
    }


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


class TestFreshnessKeyedOffStampedAt:
    """Freshness comes from ``done_provenance.stamped_at`` when present.

    This is the whole point of task 3576: ``updatedAt`` is bumped by ANY
    write to the task, so an old stamp touched later for an unrelated
    reason (a re-tag, the audit's own --apply annotation, a dependency
    edit) re-enters the window forever — the mechanical loop that keeps the
    2683 soak gate at EXIT 1 on every re-run.
    """

    def test_old_stamp_touched_after_since_is_excluded(self):
        # THE regression this task exists to kill: updatedAt says "fresh",
        # stamped_at says the attribution was asserted months ago.
        report = _report([_detail('300', 'misattributed')])
        tasks = [_stamped_task('300', AFTER_SINCE, BEFORE_SINCE)]
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_fresh_stamp_on_otherwise_untouched_task_is_included(self):
        # The converse: the stamp itself is new even though updatedAt
        # (whatever bumped it) reads as older.
        report = _report([_detail('301', 'misattributed')])
        tasks = [_stamped_task('301', BEFORE_SINCE, AFTER_SINCE)]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert [o['task_id'] for o in offenders] == ['301']

    def test_legacy_blob_without_stamped_at_falls_back_to_updated_at(self):
        # Back-compat: ~193 historical stamps carry no stamped_at at all.
        report = _report([_detail('302', 'misattributed')])
        tasks = [_stamped_task('302', AFTER_SINCE, None)]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert [o['task_id'] for o in offenders] == ['302']

    def test_unparseable_stamped_at_falls_back_to_updated_at(self):
        # The never-raises contract: a corrupt stamp degrades to the old
        # behaviour rather than blowing up the predicate.
        report = _report([_detail('303', 'misattributed')])
        tasks = [_stamped_task('303', AFTER_SINCE, 'not-a-timestamp')]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert [o['task_id'] for o in offenders] == ['303']

    def test_unparseable_stamped_at_with_old_updated_at_still_excluded(self):
        report = _report([_detail('304', 'misattributed')])
        tasks = [_stamped_task('304', BEFORE_SINCE, 'not-a-timestamp')]
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_metadata_as_json_string_is_handled(self):
        # get_tasks() hands back metadata as a JSON string in some paths;
        # parse_metadata accepts both shapes and so must this.
        report = _report([_detail('305', 'misattributed')])
        tasks = [_stamped_task('305', AFTER_SINCE, BEFORE_SINCE, as_json=True)]
        assert find_spurious_since(report, tasks, SINCE) == []

    def test_malformed_metadata_does_not_raise(self):
        report = _report([_detail('306', 'misattributed')])
        tasks = [{'id': '306', 'updatedAt': AFTER_SINCE, 'metadata': '{not json'}]
        offenders = find_spurious_since(report, tasks, SINCE)
        assert [o['task_id'] for o in offenders] == ['306']

    def test_stamped_at_exactly_at_since_is_excluded(self):
        # Strictly-after semantics, matching the existing updatedAt path.
        report = _report([_detail('307', 'misattributed')])
        tasks = [_stamped_task('307', AFTER_SINCE, '2026-07-16T00:00:00Z')]
        assert find_spurious_since(report, tasks, SINCE) == []


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


def _install_fake_audit_module(monkeypatch, report, *, ref_calls: list[str] | None = None):
    """Install a fake sibling audit module. When *ref_calls* is given, each
    call's ``ref`` kwarg is appended to it — lets a test assert args.ref is
    actually threaded into build_audit_report rather than dropped/hardcoded."""

    async def _fake_build_audit_report(tasks, git, ref='main'):
        if ref_calls is not None:
            ref_calls.append(ref)
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
    contract — exit 0 (clean) and exit 1 (offenders present).

    `since` is passed in already parsed (matching _run()'s signature: main()
    owns parsing --since — see TestMainScopedValueErrorHandling below for
    the regression guard on that split)."""

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
        exit_code = await _mod._run(args, SINCE)

        assert exit_code == 0
        backend = backend_holder['backend']
        assert backend.started is True
        assert backend.closed is True
        assert backend.get_tasks_calls == ['/proj']

    async def test_exit_one_when_post_since_misattributed_stamp_present(
        self, monkeypatch, capsys,
    ):
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
        exit_code = await _mod._run(args, SINCE)

        assert exit_code == 1
        backend = backend_holder['backend']
        assert backend.started is True
        assert backend.closed is True  # closed even on the non-zero exit path

        # Offender summary lines print to stdout on the exit-1 path —
        # human/log triage only; the DeterministicRunner parses the exit
        # code, never this text (see module docstring "Contract").
        captured = capsys.readouterr()
        assert 'task_id=8888' in captured.out
        assert f'commit={"d" * 40}' in captured.out
        assert 'flag_class=misattributed' in captured.out

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
        exit_code = await _mod._run(args, SINCE)

        assert exit_code == 1
        assert created == []

    async def test_ref_argument_is_threaded_into_build_audit_report(self, monkeypatch):
        # Regression guard: a fake build_audit_report that ignored `ref`
        # (as the other tests' fakes do) would pass even if _run() dropped
        # or hardcoded it. This one captures the actual kwarg it received.
        ref_calls: list[str] = []
        _install_fake_audit_module(monkeypatch, _report([]), ref_calls=ref_calls)
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithTaskmaster,
        )
        _install_fake_backend(monkeypatch, [])

        args = argparse.Namespace(
            project_root='/proj', config=None, ref='origin/main',
            since='2026-07-16T00:00:00Z',
        )
        exit_code = await _mod._run(args, SINCE)

        assert exit_code == 0
        assert ref_calls == ['origin/main']

    async def test_config_arg_sets_config_path_env_var(self, monkeypatch):
        monkeypatch.delenv('CONFIG_PATH', raising=False)
        _install_fake_audit_module(monkeypatch, _report([]))
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithTaskmaster,
        )
        _install_fake_backend(monkeypatch, [])

        args = argparse.Namespace(
            project_root='/proj', config='/path/to/fused-memory-config.yaml',
            ref='main', since='2026-07-16T00:00:00Z',
        )
        exit_code = await _mod._run(args, SINCE)

        assert exit_code == 0
        assert os.environ['CONFIG_PATH'] == '/path/to/fused-memory-config.yaml'


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
# main()'s ValueError->exit-2 mapping is scoped tightly around parse_since
# only — a ValueError raised later (inside _run(), e.g. from
# build_audit_report) must propagate uncaught, never get mislabeled as an
# "invalid --since" usage error just because it shares the same exception
# type. Regression guard for reviewer suggestion #1.
# ===========================================================================

class TestMainScopedValueErrorHandling:
    def test_internal_valueerror_propagates_uncaught_not_mapped_to_exit_2(
        self, monkeypatch,
    ):
        async def _raise_unrelated_valueerror(tasks, git, ref='main'):
            raise ValueError('boom: internal failure unrelated to --since')

        fake_mod = types.ModuleType('audit_found_on_main_provenance')
        fake_mod.build_audit_report = _raise_unrelated_valueerror  # type: ignore[attr-defined]
        fake_mod.GitFacts = _FakeGitFacts  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, 'audit_found_on_main_provenance', fake_mod)
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            _FakeFusedMemoryConfigWithTaskmaster,
        )
        backend_holder = _install_fake_backend(monkeypatch, [])
        monkeypatch.setattr(
            sys, 'argv',
            [
                'check_found_on_main_spurious_rate.py',
                '--since', '2026-07-16T00:00:00Z',  # a perfectly valid --since
                '--project-root', '/proj',
            ],
        )

        # Must raise the real ValueError straight out of main() — NOT
        # return exit code 2 (which would mean it got misidentified as a
        # malformed --since).
        with pytest.raises(ValueError, match='boom: internal failure'):
            _mod.main()

        # The backend's try/finally close() still runs even though the
        # exception propagates past it.
        assert backend_holder['backend'].closed is True


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
